import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



!pip install python_speech_features

from python_speech_features import logfbank
train_df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float16})

train_df['quake_id'] = (train_df.time_to_failure.diff() > 0.0).cumsum().astype('int16')

train_df.head()
lfbanks = []

quake_cnt = 15

for quake_id in range(1, quake_cnt + 1):

    quake_df = train_df.loc[train_df.quake_id == quake_id]

    burst_id = (quake_df.time_to_failure.diff() < -1e-4).cumsum().astype('int16')

    lfbank = quake_df.groupby(burst_id).apply(lambda x: logfbank(

        x.acoustic_data.values, samplerate=4096, winlen=1.0, winstep=1.0, nfilt=32, nfft=4096, highfreq=512

    ))

    lfbank = np.vstack(lfbank.values)

    lfbanks.append(lfbank)
fig, axes = plt.subplots(quake_cnt, 1, figsize=(20, 4*quake_cnt))

fig.subplots_adjust(hspace=0.3)

for idx in range(1, quake_cnt + 1):

    sns.heatmap(lfbanks[idx-1].T, ax=axes[idx-1]).set_title('Quake #{}'.format(idx))