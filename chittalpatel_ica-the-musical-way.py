PATH="/kaggle/input/icamusical/"
!pip install pydub

from pydub import AudioSegment

import IPython

import numpy as np

import wave



mix_1_wave = wave.open(PATH+'ICA mix 1.wav','r')
mix_1_wave.getparams()
264515/44100
# Extract Raw Audio from Wav File

signal_1_raw = mix_1_wave.readframes(-1)

signal_1 = np.fromstring(signal_1_raw, 'Int16')
'length: ', len(signal_1) , 'first 100 elements: ',signal_1[:100]
import matplotlib.pyplot as plt



fs = mix_1_wave.getframerate()

timing = np.linspace(0, len(signal_1)/fs, num=len(signal_1))





plt.figure(figsize=(12,2))

plt.title('Recording 1')

plt.plot(timing,signal_1, c="#3ABFE7")

plt.ylim(-35000, 35000)

plt.show()


mix_2_wave = wave.open(PATH+'ICA mix 2.wav','r')



#Extract Raw Audio from Wav File

signal_raw_2 = mix_2_wave.readframes(-1)

signal_2 = np.fromstring(signal_raw_2, 'Int16')





mix_3_wave = wave.open(PATH+'ICA mix 3.wav','r')



#Extract Raw Audio from Wav File

signal_raw_3 = mix_3_wave.readframes(-1)

signal_3 = np.fromstring(signal_raw_3, 'Int16')





plt.figure(figsize=(12,2))

plt.title('Recording 2')

plt.plot(timing,signal_2, c="#3ABFE7")

plt.ylim(-35000, 35000)

plt.show()





plt.figure(figsize=(12,2))

plt.title('Recording 3')

plt.plot(timing,signal_3, c="#3ABFE7")

plt.ylim(-35000, 35000)

plt.show()

IPython.display.Audio(PATH+"ICA mix 1.wav")
IPython.display.Audio(PATH+"ICA mix 2.wav")
IPython.display.Audio(PATH+"ICA mix 3.wav")
X = list(zip(signal_1, signal_2, signal_3))



# Let's peak at what X looks like

X[:10]
from sklearn.decomposition import FastICA



# Initializing FastICA with n_components=3

ica = FastICA(n_components=3)



# Running the FastICA algorithm using fit_transform on dataset X

ica_result = ica.fit_transform(X)
ica_result.shape
result_signal_1 = ica_result[:,0]

result_signal_2 = ica_result[:,1]

result_signal_3 = ica_result[:,2]
# Independent Component #1

plt.figure(figsize=(12,2))

plt.title('Independent Component #1')

plt.plot(result_signal_1, c="#df8efd")

plt.ylim(-0.010, 0.010)

plt.show()



# Independent Component #2

plt.figure(figsize=(12,2))

plt.title('Independent Component #2')

plt.plot(result_signal_2, c="#87de72")

plt.ylim(-0.010, 0.010)

plt.show()



# Independent Component #3

plt.figure(figsize=(12,2))

plt.title('Independent Component #3')

plt.plot(result_signal_3, c="#f65e97")

plt.ylim(-0.010, 0.010)

plt.show()



from scipy.io import wavfile



# Converting to int, mapping the appropriate range, and increasing the volume a little bit

result_signal_1_int = np.int16(result_signal_1*32767*100)

result_signal_2_int = np.int16(result_signal_2*32767*100)

result_signal_3_int = np.int16(result_signal_3*32767*100)





# Writing wave files

wavfile.write("result_signal_1.wav", fs, result_signal_1_int)

wavfile.write("result_signal_2.wav", fs, result_signal_2_int)

wavfile.write("result_signal_3.wav", fs, result_signal_3_int)
IPython.display.Audio("result_signal_1.wav")
IPython.display.Audio("result_signal_3.wav")
IPython.display.Audio("result_signal_2.wav")