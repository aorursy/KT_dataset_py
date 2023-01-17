!pip install nussl
import nussl

import IPython

import warnings

import IPython.display as ipd

import matplotlib.pyplot as plt



path = "../input/birdsong-recognition/train_audio/aldfly/XC135454.mp3"



history = nussl.AudioSignal(path)

history.embed_audio()



plt.figure(figsize=(10, 3))

nussl.utils.visualize_spectrogram(history)

plt.title(str(history))

plt.tight_layout()

plt.show()
repet = nussl.separation.primitive.Repet(history)

estimates = repet()

repet.repeating_period
_estimates = {

    'Background': estimates[0],

    'Foreground': estimates[1]

} # organize estimates into a dict



plt.figure(figsize=(10, 7))

plt.subplot(211)

nussl.utils.visualize_sources_as_masks(

    _estimates, db_cutoff=-60, y_axis='mel')

plt.subplot(212)

nussl.utils.visualize_sources_as_waveform(

    _estimates, show_legend=False)

plt.tight_layout()

plt.show()



nussl.play_utils.multitrack(_estimates)
def foreground(data):

    """

    params: data 1D numpy array of raw audio signal

    returns: 1D numpy array of raw audio signal with background noise removed

    """

    history = nussl.AudioSignal(path_to_input_file=None, audio_data_array=data)

    estimates = nussl.separation.primitive.Repet(history)()

    return estimates[1].audio_data[0]
path = "../input/birdsong-recognition/train_audio/amerob/XC128490.mp3"
import librosa



data, SR = librosa.core.load(path, sr=None, duration=40)
librosa.output.write_wav('before.wav', data, SR, norm=False)

history = nussl.AudioSignal('before.wav')

history.embed_audio()



plt.figure(figsize=(10, 3))

nussl.utils.visualize_spectrogram(history)

plt.title(str(history))

plt.tight_layout()

plt.show()
librosa.output.write_wav('after.wav', foreground(data), SR, norm=False)

history = nussl.AudioSignal('after.wav')

history.embed_audio()



plt.figure(figsize=(10, 3))

nussl.utils.visualize_spectrogram(history)

plt.title(str(history))

plt.tight_layout()

plt.show()