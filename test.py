import datetime
import os

from dotenv import load_dotenv
from nnsight import CONFIG, LanguageModel

load_dotenv()


NDIF_API_KEY = os.getenv('NDIF_API_KEY')
CONFIG.set_default_api_key(NDIF_API_KEY)

SCHEDULE = {
    'Monday': 'Base',
    'Tuesday': 'Base',
    'Wednesday': 'Base',
    'Thursday': 'Base',
    'Friday': 'Instruct',
    'Saturday': 'Instruct',
    'Sunday': 'Instruct',
}

today = datetime.datetime.today().strftime('%A')
expected_mode = SCHEDULE[today]

user_mode = input('Enter mode (Base/Instruct): ').strip().title()

if user_mode != expected_mode:
    raise ValueError(
        f"Error: Today is {today}, expected mode is '{expected_mode}', "
        f"but user selected '{user_mode}'."
    )

print(f'Running in {user_mode} mode...')

model_name = (
    'meta-llama/Meta-Llama-3.1-8B'
    if user_mode == 'Base'
    else 'meta-llama/Meta-Llama-3.1-8B-Instruct'
)

# Initialize the LanguageModel.
llama = LanguageModel(model_name)

# Use the tracing context with remote execution.
with llama.trace('The Eiffel Tower is in the city of', remote=True):

    # Inside the tracing context, access and save desired model internals.
    hidden_states = llama.model.layers[-1].output.save()
    output = llama.output.save()

print(hidden_states)
print(output['logits'])
