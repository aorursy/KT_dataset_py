import pandas as pd
import numpy as np
from scipy import fftpack
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
%matplotlib inline
train = pd.read_csv('../input/train.csv', nrows=10000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print("train shape", train.shape)
train.head()
sampling_rate = 4000000
def log_specgram(data, sample_rate, nperseg=2000, noverlap=1000, eps=1e-10, dct=False):
    freqs, times, spec = signal.spectrogram(data,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    spec = np.log(spec).astype(np.float32)
    if dct:
        spec = fftpack.dct(spec, type=2, axis=0, norm='ortho')
    return freqs, times, spec


def plot_specgram(data, sample_rate, final_idx, init_idx=0, step=1, nperseg=2000, 
                  noverlap=1000, dct=False, title='', subsampling=False):
    idx = [i for i in range(init_idx, final_idx, step)]
    acoustic_data = data.iloc[idx].acoustic_data.values
    if subsampling:
        acoustic_data = acoustic_data[::subsampling]
    freqs, times, spectrogram = log_specgram(acoustic_data, sample_rate, 
                                             nperseg=nperseg, noverlap=noverlap, dct=dct)

    plt.figure(figsize=(10, 8))
    plt.imshow(spectrogram, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    plt.title(title)
    plt.ylabel('Freqs in Hz')
    plt.xlabel('Seconds')
    plt.show()
def single_timeseries(final_idx, init_idx=0, step=1, title="",
                      color1='orange', color2='blue', subsampling=False):
    idx = [i for i in range(init_idx, final_idx, step)]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14)
    
    ax2 = ax1.twinx()
    ax1.set_xlabel('index')
    ax1.set_ylabel('Acoustic data')
    ax2.set_ylabel('Time to failure')
    acoustic_data = train.iloc[idx].acoustic_data.values
    time_to_failure = train.iloc[idx].time_to_failure.values
    if subsampling:
        acoustic_data = acoustic_data[::subsampling]
        time_to_failure = time_to_failure[::subsampling]

    p1 = sns.lineplot(data=acoustic_data, ax=ax1, color=color1)
    p2 = sns.lineplot(data=time_to_failure, ax=ax2, color=color2)

plot_specgram(train, sampling_rate, 100000, title='Specgram of first hundred thousand rows')
single_timeseries(100000, title="First hundred thousand rows")
plot_specgram(train, sampling_rate, final_idx=6000000, init_idx=5000000, 
              title='Specgram of earthquake')
single_timeseries(final_idx=6000000, init_idx=5000000, title="Five to six million index")
plot_specgram(train, sampling_rate, 10000, title='Specgram of first hundred thousand rows', dct=True)
single_timeseries(80000, title="First eighty thousand rows")
plot_specgram(train, sampling_rate, 80000, title='Specgram of first eighty thousand rows')
single_timeseries(80000, title="First eighty thousand rows - subsampled", subsampling=4)

plot_specgram(train, sample_rate=1000000, final_idx=80000, nperseg=500, noverlap=250, 
              title='Subsampled specgram of first eighty thousand rows', subsampling=4)