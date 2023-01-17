import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.signal

%matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 15.0)

print('Reading train data')
df_train = pd.read_csv('../input/train.csv')

print('\nSize of training data: ' + str(df_train.shape))
print('Columns:' + str(df_train.columns.values))
print('Number of places: ' + str(len(list(set(df_train['place_id'].values.tolist())))))
print('\n')
# Pull out timestamps
times = np.squeeze(df_train.as_matrix(columns=['time']))

n_events = times.size
n_samples = n_events
hist_range = (-100000.0, 100000.0)
n_bins = 100000 # One bin per 2.0 time units
n_loops = 100

# Estimate autocorrelations
# Randomly pull timestamps and subtract from each other. Get histogram of these values toestimate autocorrelation
print('Estimating autocorrelation')
all_autocorrs_global = np.zeros((n_bins, n_loops))
for loop_n in range(n_loops):
  hist_vals, bin_edges = np.histogram(np.random.choice(times, size=n_samples, replace=True) - \
                                      np.random.choice(times, size=n_samples, replace=True), bins=n_bins, range=hist_range)
  all_autocorrs_global[:, loop_n] = hist_vals

# Plot the autocorrelation and fft of autocorrelation
fig, axs = plt.subplots(2,1)

axs[0].plot(bin_edges[:-1], all_autocorrs_global)
axs[0].set_xlim([-20000.0, 20000.0])
axs[0].set_title('Autocorrelation of all timestamps')
axs[0].set_xlabel('time units')

# Plot the fft of the autocorrelation
f, psd = scipy.signal.welch(all_autocorrs_global, nperseg=25000, noverlap=20000, return_onesided=True, axis=0)

# Adjust the X axis to be in time points instead of 1/F
f /= 2.0  # Remember that there is one bin per 4.0 time units
f = 1.0/f # Go back to time points

# Plot the fft of the autocorrelation (The PSD of the timestamps)
axs[1].plot(f, np.log(psd))
axs[1].set_title('Log FFT of global autocorrelations')
axs[1].set_xlabel('time units')
axs[1].set_xlim([0.0, 20000.0])
axs[1].set_xticks(np.arange(0.0, 20000.0, 1000))
axs[1].grid(True)

fig.tight_layout()

plt.show()
# Get places with most check-ins
places_by_frequency = df_train.groupby('place_id')['place_id'].agg('count').sort_values(ascending=False).index.tolist()
n_places_to_analyze = 100
places_by_frequency = places_by_frequency[:n_places_to_analyze]


hist_range = (-100000.0, 100000.0)
n_bins = 50000 # One bin per 4.0 time units

# Get the autocorrelation between timestamps for each place
all_autocorrs = np.zeros((n_bins, n_places_to_analyze))
place_n = 0
for place_id in places_by_frequency:
  times = np.squeeze(df_train[df_train['place_id']==place_id].as_matrix(columns=['time']))
  n_events = times.size
  n_samples = n_events*n_events # We are still randomly choosing timestamps, but this should give good coverage
  hist_vals, bin_edges = np.histogram(np.random.choice(times, size=n_samples, replace=True) - \
                                    np.random.choice(times, size=n_samples, replace=True), bins=n_bins,
                                    range=hist_range)
  all_autocorrs[:, place_n] = hist_vals
  place_n += 1


# Plot the autocorrelation and fft of autocorrelation
fig, axs = plt.subplots(3,1)

axs[0].plot(bin_edges[:-1], all_autocorrs)
axs[0].set_title('Autocorrelations for each place')
axs[0].set_xlabel('time units')
axs[0].set_xlim([-20000, 20000.0])
axs[0].set_ylim([0.0, 200.0])


f, psd = scipy.signal.welch(all_autocorrs, nperseg=25000, noverlap=20000, return_onesided=True, axis=0)

# Adjust the X axis to be in time points instead of 1/F
f /= 4.0 # Remember that there is one bin per 4.0 time units
f = 1.0/f # Go back to time points

# Plot the fft of the autocorrelation (The PSD of the timestamps)
axs[1].plot(f, np.log(psd))
axs[1].set_title('Log FFT of autocorrelations for each place')
axs[1].set_xlabel('time units')
axs[1].set_xlim([0.0, 40000.0])
axs[1].set_xticks(np.arange(0.0, 40000.0, 2000))
axs[1].grid(True)

# Plot the fft of the autocorrelation (zoomed)
axs[2].plot(f, np.log(psd))
axs[2].set_title('Log FFT of autocorrelations for each place')
axs[2].set_xlabel('time units')
axs[2].set_xlim([0.0, 2500.0])
axs[2].set_xticks(np.arange(0.0, 2500.0, 100))
axs[2].grid(True)

fig.tight_layout()
plt.show()
