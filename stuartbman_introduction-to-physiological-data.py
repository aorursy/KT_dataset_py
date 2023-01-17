import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv('../input/train.csv')

# Just looking at a single trial for now
subset = df_train.loc[(df_train['crew'] == 1) & (df_train['experiment'] == 'CA')]

subset.sort_values(by='time')


# Show the plot
plt.plot(subset['r'][3000:4024])

from scipy import signal

b, a = signal.butter(8,0.05)

y = signal.filtfilt(b, a, subset['r'], padlen=150)

plt.plot(y[3000:4024])
from biosppy.signals import ecg, resp

out = resp.resp(y,sampling_rate=256, show=False)

plt.plot(out['resp_rate_ts'], out['resp_rate'])
plt.ylabel('Respiratory frequency [Hz]')
plt.xlabel('Time [s]');
b, a = signal.butter(8,0.05)

y = signal.filtfilt(b, a, subset['ecg'], padlen=150)

plt.plot(y[3000:4024])

out = ecg.ecg(signal=subset['ecg'], sampling_rate=256, show=False)

plt.plot(out['heart_rate_ts'], out['heart_rate'])
plt.ylabel('Heart Rate (BPM)')
plt.xlabel('Time [s]');
df_train['fp1_f7'] = df_train['eeg_fp1'] - df_train['eeg_f7']
df_train['f7_t3'] = df_train['eeg_f7'] - df_train['eeg_t3']
df_train['t3_t5'] = df_train['eeg_t3'] - df_train['eeg_t5']
df_train['t5_o1'] = df_train['eeg_t5'] - df_train['eeg_o1']
df_train['fp1_f3'] = df_train['eeg_fp1'] - df_train['eeg_f7']
df_train['f3_c3'] = df_train['eeg_f3'] - df_train['eeg_c3']
df_train['c3_p3'] = df_train['eeg_c3'] - df_train['eeg_p3']
df_train['p3_o1'] = df_train['eeg_p3'] - df_train['eeg_o1']

df_train['fz_cz'] = df_train['eeg_fz'] - df_train['eeg_cz']
df_train['cz_pz'] = df_train['eeg_cz'] - df_train['eeg_pz']
df_train['pz_poz'] = df_train['eeg_pz'] - df_train['eeg_poz']

df_train['fp2_f8'] = df_train['eeg_fp2'] - df_train['eeg_f8']
df_train['f8_t4'] = df_train['eeg_f8'] - df_train['eeg_t4']
df_train['t4_t6'] = df_train['eeg_t4'] - df_train['eeg_t6']
df_train['t6_o2'] = df_train['eeg_t6'] - df_train['eeg_o2']
df_train['fp2_f4'] = df_train['eeg_fp2'] - df_train['eeg_f4']
df_train['f4_c4'] = df_train['eeg_f4'] - df_train['eeg_c4']
df_train['c4_p4'] = df_train['eeg_c4'] - df_train['eeg_p4']
df_train['p4_o2'] = df_train['eeg_p4'] - df_train['eeg_o2']

features_n = ['fp1_f7', 'f7_t3', 't3_t5', 't5_o1', 'fp1_f3', 'f3_c3', 'c3_p3', 'p3_o1', 'fz_cz', 'cz_pz',
                'pz_poz', 'fp2_f8', 'f8_t4', 't4_t6', 't6_o2', 'fp2_f4', 'f4_c4', 'c4_p4', 'p4_o2', "ecg", "r", "gsr"]

subset = df_train.loc[(df_train['crew'] == 1)]

# Discrete Fourier transform, using a hanning window of 1s
freqs, times, Sx = signal.spectrogram(subset['fz_cz'], fs=256, window='hanning', nperseg=256, noverlap=256-100, detrend=False, scaling='spectrum')
f, ax = plt.subplots(figsize=(12,5))
ax.pcolormesh(times, freqs, 10 * np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [s]');
b, a = signal.butter(8,0.2) 
y = signal.filtfilt(b, a, subset['fz_cz'], padlen=150)
freqs, times, Sx = signal.spectrogram(y, fs=256, window='hanning', nperseg=256, noverlap=256-100, detrend=False, scaling='spectrum')
f, ax = plt.subplots(figsize=(12,5))
ax.pcolormesh(times, freqs, 10 * np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [s]');