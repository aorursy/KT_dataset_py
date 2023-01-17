import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()

sns.set_style('whitegrid')



from pymssa import MSSA
# Read 20k rows and keep only acoustic data

df = pd.read_csv("../input/train.csv", nrows=20000, dtype={'acoustic_data': np.int16, 

                                                           'time_to_failure': np.float64})[['acoustic_data']]

# Zero-center data

df['acoustic_data'] -= df['acoustic_data'].mean()

df.head(5)
FIGSIZE = (16, 16)



fig, ax = plt.subplots(2, 1, figsize=FIGSIZE)

g = sns.lineplot(x=df.index[::2], y=df['acoustic_data'][::2], ax=ax[0])

ax[0].ticklabel_format(useOffset=False)

ax[0].set_title('Waveform');

ax[0].set_xlabel('Time')

ax[0].set_ylabel('Amplitude')

g = sns.lineplot(x=df.index[:100], y=df['acoustic_data'][:100], ax=ax[1])

ax[1].ticklabel_format(useOffset=False)

ax[1].set_title('Waveform 100 samples zoom');

ax[1].set_xlabel('Time')

ax[1].set_ylabel('Amplitude')

plt.show()
N_COMPONENTS = 16

mssa = MSSA(n_components=N_COMPONENTS, window_size=4096, verbose=True)
mssa.fit(df)
waveform = df['acoustic_data'].values

cumulative_recon = np.zeros_like(waveform)



for comp in range(N_COMPONENTS):  

    fig, ax = plt.subplots(figsize=(18, 7))

    current_component = mssa.components_[0, :, comp]

    cumulative_recon = cumulative_recon + current_component

    

    ax.plot(df.index, waveform, lw=3, alpha=0.2, c='k', label='waveform')

    ax.plot(df.index, cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))

    ax.plot(df.index, current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))

    

    ax.legend()

    plt.show()
waveform = df['acoustic_data'].values[:100]

cumulative_recon = np.zeros_like(waveform)



for comp in range(N_COMPONENTS):  

    fig, ax = plt.subplots(figsize=(18, 7))

    current_component = mssa.components_[0, :100, comp]

    cumulative_recon = cumulative_recon + current_component

    

    ax.plot(df.index[:100], waveform, lw=3, alpha=0.2, c='k', label='Waveform 100 samples zoom')

    ax.plot(df.index[:100], cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))

    ax.plot(df.index[:100], current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))

    

    ax.legend()

    plt.show()