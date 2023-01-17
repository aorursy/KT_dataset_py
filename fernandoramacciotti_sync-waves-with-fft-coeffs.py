import numpy as np

from scipy import fftpack # Fast Fourier Transform functions



import pyarrow.parquet as pq # for reading input data



%matplotlib inline

import matplotlib.pyplot as plt # plotting



plt.style.use('seaborn-whitegrid')
# load just three first signals (which belong to the same id measurement)

n_signals_to_load = 3

signals = pq.read_pandas(

    '../input/train.parquet', 

    columns=[str(i) for i in range(n_signals_to_load)]).to_pandas()
# sampling rate

num_samples = signals.shape[0] # 800,000 samples per signal

period = 0.02 # over a 20ms period

fs = num_samples / period # 40MHz sampling rate



# time array support

t = np.array([i / fs for i in range(num_samples)])



# frequency vector fro FFT

freqs = fftpack.fftfreq(num_samples, d=1/fs)
# get fft coeffs

def get_fft_coeffs(sig):

    return fftpack.fft(sig)



# get coeff with highest norm

def get_highest_coeff(fft_coeffs, freqs, verbose=True):

    coeff_norms = np.abs(fft_coeffs) # get norms (fft coeffs are complex)

    max_idx = np.argmax(coeff_norms)

    max_coeff = fft_coeffs[max_idx] # get max coeff

    max_freq = freqs[max_idx] # assess which is the dominant frequency

    max_amp = (coeff_norms[max_idx] / num_samples) * 2 # times 2 because there are mirrored freqs

    if verbose:

        print('Dominant frequency is {:,.1f}Hz with amplitude of {:,.1f}\n'.format(max_freq, max_amp))

    

    return max_coeff, max_amp, max_freq



# get max coeff phase

def get_max_coeff_phase(max_coeff):

    return np.angle(max_coeff)



# construct the instant angular phase vector indexed by pi, i.e. ranges from 0 to 2

def get_instant_w(time_vector, f0, phase_shift):

    w_vector = 2 * np.pi * time_vector * f0 + phase_shift

    w_vector_norm = np.mod(w_vector / (2 * np.pi), 1) * 2 # range between cycle of 0-2 

    return w_vector, w_vector_norm



# find index of chosen phase to align

def get_align_idx(w_vector_norm, align_value=0.5):

    candidates = np.where(np.isclose(w_vector_norm, align_value))

    # since we are in discrete time, threre could be many values close to the desired one

    # so let's take the one in the middle

    return int(np.median(candidates))
fig = plt.figure(figsize=(16, 9))

plot_number = 0



for signal_id in signals.columns:

    # get samples

    print('=== Signal {} ==='.format(signal_id))

    sig = signals[signal_id]

    

    # fft

    fft_coeffs = get_fft_coeffs(sig)

    

    # asses dominant frequency

    max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs)

    

    # phase shift

    ps = get_max_coeff_phase(max_coeff)

    

    # get angular phase vector

    w, w_norm = get_instant_w(t, f0, ps)

    

    # generate dominant signal at f0

    dominant_wave = amp * np.cos(w) # if np.sin(), then need to ajust by pi/2

    (w)

    

    # plot signals

    plot_number += 1

    ax = fig.add_subplot(3, 2, plot_number)

    

    ax.plot(t * 1000, sig, label='Original') # original signal

    ax.plot(t * 1000, dominant_wave, color='red', label='Wave at {:.0f}Hz'.format(f0)) # wave at f0

    ax.legend()

    ax.set_xlabel('time (ms)')

    ax.set_ylabel('Amplitude')

    ax.set_title('Signal {}'.format(signal_id))

    

    # plot phase

    plot_number += 1

    ax = fig.add_subplot(3, 2, plot_number)

    

    ax.plot(t * 1000, w_norm, label='phase') # instant phase

    ax2 = ax.twinx() # secondary y

    ax2.plot(t * 1000, dominant_wave, color='red', label='Wave at {:.0f}Hz'.format(f0)) # wave at f0

    ax.legend()

    ax.set_xlabel('time (ms)')

    ax.set_ylabel('$\omega_i (\pi$ rad)')

    ax2.set_ylabel('Wave amplitude')

    ax.set_title('Instant angular phase of dominant wave')

    

fig.tight_layout()
# align waves with np.roll()

align_phase = 0.5 # w_i = pi/2





fig = plt.figure(figsize=(12, 9))

plot_number = 0



for signal_id in signals.columns:

    # get samples

    sig = signals[signal_id]

    

    # fft

    fft_coeffs = get_fft_coeffs(sig)

    

    # asses dominant frequency

    max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs, verbose=False)

    

    # phase shift

    ps = get_max_coeff_phase(max_coeff)

    

    # get angular phase vector

    w, w_norm = get_instant_w(t, f0, ps)

    

    # generate dominant signal at f0

    dominant_wave = amp * np.cos(w)

    

    # idx to roll

    origin = get_align_idx(w_norm, align_value=align_phase)

    

    # roll signal and dominant wave

    sig_rolled = np.roll(sig, num_samples - origin)

    dominant_wave_rolled = np.roll(dominant_wave, num_samples - origin)

    

    # plot signals

    plot_number += 1

    ax = fig.add_subplot(3, 1, plot_number)

    

    ax.plot(t * 1000, sig_rolled, label='Rolled Original') # original signal

    ax.plot(t * 1000, dominant_wave_rolled, color='red', label='Rolled Wave at {:.0f}Hz'.format(f0)) # wave at f0

    ax.legend()

    ax.set_xlabel('time (ms)')

    ax.set_ylabel('Amplitude')

    ax.set_title('Signal {} rolled'.format(signal_id))

    

fig.tight_layout()
# align waves with np.roll()

align_phase = 0.25 # w_i = pi/4





fig = plt.figure(figsize=(12, 9))

plot_number = 0



for signal_id in signals.columns:

    # get samples

    sig = signals[signal_id]

    

    # fft

    fft_coeffs = get_fft_coeffs(sig)

    

    # asses dominant frequency

    max_coeff, amp, f0 = get_highest_coeff(fft_coeffs, freqs, verbose=False)

    

    # phase shift

    ps = get_max_coeff_phase(max_coeff)

    

    # get angular phase vector

    w, w_norm = get_instant_w(t, f0, ps)

    

    # generate dominant signal at f0

    dominant_wave = amp * np.cos(w)

    

    # idx to roll

    origin = get_align_idx(w_norm, align_value=align_phase)

    

    # roll signal and dominant wave

    sig_rolled = np.roll(sig, num_samples - origin)

    dominant_wave_rolled = np.roll(dominant_wave, num_samples - origin)

    

    # plot signals

    plot_number += 1

    ax = fig.add_subplot(3, 1, plot_number)

    

    ax.plot(t * 1000, sig_rolled, label='Rolled Original') # original signal

    ax.plot(t * 1000, dominant_wave_rolled, color='red', label='Rolled Wave at {:.0f}Hz'.format(f0)) # wave at f0

    ax.legend()

    ax.set_xlabel('time (ms)')

    ax.set_ylabel('Amplitude')

    ax.set_title('Signal {} rolled'.format(signal_id))

    

fig.tight_layout()