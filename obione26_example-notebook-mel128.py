import glob

import numpy as np

import librosa

import librosa.display

import IPython

import IPython.display as ipd
def load_random(paths):

    mel_path = np.random.choice(paths, 1)[0]

    mel = np.load(mel_path)

    return mel



def display_spectra(mel):

    log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)

    librosa.display.specshow(log_mel, sr=44100, x_axis='time', y_axis='log', hop_length=347, fmin=20, fmax=44100//2)



def convert_spectra2sound(mel):

    sample = librosa.feature.inverse.mel_to_audio(mel, sr=44100, n_fft=20*128, hop_length=347)

    return sample

        

def listen_sample(sample, sr=44100):

    return IPython.display.display(ipd.Audio(data=sample, rate=sr))
curated_paths = glob.glob('../input/mel128/train_curated_mel/*.npy')

noisy_paths = glob.glob('../input/mel128/train_noisy_mel/*.npy')

test_paths = glob.glob('../input/mel128/test_mel/*.npy')
# Dispay random 

mel = load_random(noisy_paths)

display_spectra(mel)

print(f'Mel: {mel.shape}')
# Reconstruct sample signal from mel spectrogram

sample = convert_spectra2sound(mel)

listen_sample(sample)

print(f'Sample - {sample.shape}')