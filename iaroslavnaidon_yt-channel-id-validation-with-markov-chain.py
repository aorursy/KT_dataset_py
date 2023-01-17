!pip install striplog

# source code: https://github.com/agile-geoscience/striplog/blob/78ab7dbb17ab0589f973b34d939a1271a16b98a9/striplog/markov.py
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



csv_file = '/kaggle/input/youtube-channels-100000/channels.csv'

data =  pd.read_csv(csv_file)

channels = data['channel_id'].tolist()



data.head()
from striplog.markov import Markov_chain



# add start and end symbol markers

START_MARKER = '<'

END_MARKER = '>'

marked_channels = [START_MARKER + item + END_MARKER for item in channels]



m = Markov_chain.from_sequence(marked_channels, strings_are_states=False, include_self=True, step=1)

m.plot_norm_diff()
from striplog.markov import Markov_chain



# add start and end symbol markers

START_MARKER = '<'

END_MARKER = '>'



# split each channel id like 'UC_test' into ['0-U', '1-C', '2-_', '3-t', '4-e', '5-s', '6-t']

# to make 'position-symbol' represent a single transition state

states = []

for channel_id in channels:

    states.append(

        [START_MARKER] +

        [str(idx) + '-' + ch for idx,ch in enumerate(channel_id)] +

        [END_MARKER]

    )



m = Markov_chain.from_sequence(states, strings_are_states=True, include_self=False, step=1)

# m.plot_norm_diff() # this causes Out of Memory error
def find_max_probable_transition_state(state_from):

    idx_from = m._index_dict[state_from]

    max_prob = 0

    state_to = None

    for idx_to,prob in enumerate(m.observed_freqs[idx_from]):

        if (prob > max_prob):

            max_prob = prob

            state_to = m.states[idx_to]



    return (state_to, max_prob)



anomaly_threshold = 0.3



df_anomalies = pd.DataFrame(columns=['From', 'To', 'Probability'])



for state in m.states:

    state_to, prob = find_max_probable_transition_state(state)

    if (prob > anomaly_threshold):

        df_anomalies.loc[len(df_anomalies)] = [state, state_to, prob]



df_anomalies
def get_channel_validity_score(channel_id):

    channel_id = [str(idx) + '-' + ch for idx,ch in enumerate(channel_id)] + [END_MARKER]

    score = 0

    transition_probs = m.observed_freqs[m._index_dict[START_MARKER]]

    for idx, ch in enumerate(channel_id):

        # exit if the character was not present in the training data

        if ch not in m.states:

            return 0

        

        ch_prob = transition_probs[m._index_dict[ch]]

        # exit if the character transition is not probable

        if ch_prob == 0:

            return 0

        score += (1 + ch_prob)

        

        transition_probs = m.observed_freqs[m._index_dict[ch]]



    return score



def is_valid(channel_id):

    score = get_channel_validity_score(channel_id)

    return score > 0
assert not is_valid('hello_world')

assert not is_valid('playlist?list=ELMd5hEoIMJHtaXbMgH4pSzA')

assert not is_valid('ddMd5hEoIMJHtaXbMgH4pSzA')

assert not is_valid('KC_oorT0w_bwkl95l3LoKUzw') # doesn't start with U

assert not is_valid('UD_oorT0w_bwkl95l3LoKUzw') # doesn't start with UC

assert not is_valid('UC_oorT0w_bwkl95l3LoKUza') # doesn't end with 'A', 'Q', 'g', or 'w'

assert not is_valid('UC_oorT0w_bwkl95l3LoKUzzw') # wrong length - too long

assert not is_valid('UC_oorT0w_bwkl95l3LoKUw') # wrong length - too short



assert is_valid('UC_oorT0w_bwkl95l3LoKUzw')

assert is_valid('UCKrrATalRpJ-H5ltP788DkA')

assert is_valid('UCIU8ha-NHmLjtUwU7dFiXUA')

assert is_valid('UClK-ywlLUC-_EUC1qsIiOfA')

csv_file = '/kaggle/input/youtube-channels-100000/channels.csv'

data =  pd.read_csv(csv_file).sample(1000)

data['length'] = data['channel_id'].str.len()

data['validity_score'] = data['channel_id'].apply(lambda x: get_channel_validity_score(x))

data['is_valid'] = data['channel_id'].apply(lambda x: is_valid(x))



data[['channel_id','length','validity_score','is_valid']].head()

assert len(data.loc[data['is_valid'] == False]) == 0

csv_file = '/kaggle/input/youtube-top-5000-channel-ids/output.csv'

data = pd.read_csv(csv_file).sample(1000)

data['length'] = data[' ID'].str.len()

data['validity_score'] = data[' ID'].apply(lambda x: get_channel_validity_score(x))

data['is_valid'] = data[' ID'].apply(lambda x: is_valid(x))

data.head()
assert len(data.loc[data['is_valid'] == False]) == 0
import re



pat = re.compile('(UC[A-Za-z0-9\-_]{21}[AQgw])')



unmatched = [channel for channel in channels if not pat.match(channel)]



assert len(unmatched) == 0