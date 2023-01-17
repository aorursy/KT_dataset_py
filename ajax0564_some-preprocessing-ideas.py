import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import librosa

import librosa.display

%matplotlib inline
audio_path = '../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'

x , sr = librosa.load(audio_path)
plt.figure(figsize=(16, 4))

librosa.display.waveplot(x, sr=sr)

plt.title('Slower Version $X_1$')

plt.tight_layout()

plt.show()
S = librosa.feature.melspectrogram(x, sr=sr, n_mels=128)



# Convert to log scale (dB). We'll use the peak power (max) as reference.

log_S = librosa.power_to_db(S, ref=np.max)



# Make a new figure

plt.figure(figsize=(12,4))



# Display the spectrogram on a mel scale

# sample rate and hop length parameters are used to render the time axis

librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')



# Put a descriptive title on the plot

plt.title('mel power spectrogram')



# draw a color bar

plt.colorbar(format='%+02.0f dB')



# Make the figure layout compact

plt.tight_layout()
import IPython.display as ipd

ipd.Audio(audio_path)
y_harmonic, y_percussive = librosa.effects.hpss(x)


C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)



# Make a new figure

plt.figure(figsize=(12,4))



# Display the chromagram: the energy in each chromatic pitch class as a function of time

# To make sure that the colors span the full range of chroma values, set vmin and vmax

librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)



plt.title('Chromagram')

plt.colorbar()



plt.tight_layout()
ipd.Audio(data=y_harmonic, rate=sr)
ipd.Audio(data=y_percussive, rate=sr)
y_shift = librosa.effects.pitch_shift(x, sr, 7)



ipd.Audio(data=y_shift, rate=sr)
D = librosa.stft(x)



# Separate the magnitude and phase

S, phase = librosa.magphase(D)



# Decompose by nmf

components, activations = librosa.decompose.decompose(S, n_components=8, sort=True)





plt.figure(figsize=(12,4))



plt.subplot(1,2,1)

librosa.display.specshow(librosa.amplitude_to_db(np.abs(components), ref=np.max), y_axis='log')

plt.xlabel('Component')

plt.ylabel('Frequency')

plt.title('Components')



plt.subplot(1,2,2)

librosa.display.specshow(activations, x_axis='time')

plt.xlabel('Time')

plt.ylabel('Component')

plt.title('Activations')



plt.tight_layout()
# we isolate just last (highest) component?

k = -1



# Reconstruct a spectrogram by the outer product of component k and its activation

D_k = np.multiply.outer(components[:, k], activations[k])



# invert the stft after putting the phase back in

y_k = librosa.istft(D_k * phase)



# And playback

print('Component #{}'.format(k))



ipd.Audio(data=y_k, rate=sr)
def  apply_tunning(y):

    '''Load audio, estimate tuning, apply pitch correction, and save.'''

  



    print('Separating harmonic component ... ')

    y_harm = librosa.effects.harmonic(y)



    print('Estimating tuning ... ')

    # Just track the pitches associated with high magnitude

    tuning = librosa.estimate_tuning(y=y_harm, sr=sr)



    print('{:+0.2f} cents'.format(100 * tuning))

    print('Applying pitch-correction of {:+0.2f} cents'.format(-100 * tuning))

    y_tuned = librosa.effects.pitch_shift(y, sr, -tuning)

    return  y_tuned

y_tuned = apply_tunning(x)
y_tuned
# Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)

S = librosa.feature.melspectrogram(x, sr=sr, n_mels=128)



# Convert to log scale (dB). We'll use the peak power (max) as reference.

log_S = librosa.power_to_db(S, ref=np.max)

mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=15)



# Let's pad on the first and second deltas while we're at it

delta_mfcc  = librosa.feature.delta(mfcc)

delta2_mfcc = librosa.feature.delta(mfcc, order=2)



# How do they look?  We'll show each in its own subplot

plt.figure(figsize=(12, 6))



plt.subplot(3,1,1)

librosa.display.specshow(mfcc)

plt.ylabel('MFCC')

plt.colorbar()



plt.subplot(3,1,2)

librosa.display.specshow(delta_mfcc)

plt.ylabel('MFCC-$\Delta$')

plt.colorbar()



plt.subplot(3,1,3)

librosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')

plt.ylabel('MFCC-$\Delta^2$')

plt.colorbar()



plt.tight_layout()

S_full, phase = librosa.magphase(librosa.stft(x, hop_length=2048, window=np.ones))
idx = slice(*librosa.time_to_frames([10, 15], hop_length=2048, sr=sr))

plt.figure(figsize=(12, 4))

librosa.display.specshow(librosa.power_to_db(S_full[:, idx]**2, ref=np.max),

                         y_axis='log', x_axis='time', sr=sr)

plt.colorbar()

plt.tight_layout()
S_filter = librosa.decompose.nn_filter(S_full,

                                       aggregate=np.median,

                                       metric='cosine',

                                       k=20,

                                       width=int(librosa.time_to_frames(2, sr=sr)))



# The output of the filter shouldn't be greater than the input if we assume signals are additive

S_filter = np.minimum(S_full, S_filter)
# Note: the margins need not be equal for foreground and background separation

margin_i, margin_v = 2, 10

power = 2



mask_i = librosa.util.softmask(S_filter,

                               margin_i * (S_full - S_filter),

                               power=power)



mask_v = librosa.util.softmask(S_full - S_filter,

                               margin_v * S_filter,

                               power=power)



# Using the masks we get a cleaner signal



S_foreground = mask_v * S_full

S_background = mask_i * S_full
plt.figure(figsize=(12, 8))

plt.subplot(2,1,1)

librosa.display.specshow(librosa.power_to_db(S_background[:, idx]**2, ref=np.max),

                         y_axis='log', sr=sr)

plt.title('Background')

plt.colorbar()

plt.subplot(2,1,2)

librosa.display.specshow(librosa.power_to_db(S_foreground[:, idx]**2, ref=np.max),

                         y_axis='log', x_axis='time', sr=sr)

plt.title('Foreground')

plt.colorbar()

plt.tight_layout()
ipd.Audio(data=librosa.istft(S_background * phase), rate=sr)
ipd.Audio(data=librosa.istft(S_foreground * phase), rate=sr)