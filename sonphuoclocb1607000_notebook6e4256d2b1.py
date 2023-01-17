import os

import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

root_data = '../input/data-wav/'

#DFT - Discrete Fourier Transform

def plot_spectrogram(Y, sr, hop_length, save_fig, y_axis):

    plt.figure(figsize = (25, 10))

    librosa.display.specshow(Y, hop_length = hop_length, x_axis = 'time', y_axis = y_axis)

    plt.colorbar(format = '%+2.f')

#     plt.savefig(save_fig + '.png')

#     plt.show()

for subdirs, dirs, files in os.walk(root_data):

    for file in files:

        extenstion = os.path.splitext(file)[-1].lower()  

        if extenstion == '.wav':

            file_path_output = subdirs + file.replace(extenstion, '') + '.wav'

            scale, sr = librosa.load(file_path_output)

            #Short-time Fourier transform (STFT).

            #The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.

            FRAME_SIZE = 2048

            HOP_SIZE = 512

            S_scale = librosa.stft(scale, n_fft = FRAME_SIZE, hop_length = HOP_SIZE)

            print(S_scale.shape)

            print(type(S_scale[0][0]))

            #Calculating the spectrogram

            Y_scale = np.abs(S_scale) ** 2

            print(Y_scale.shape)

            print(type(Y_scale[0][0]))

#             plot_spectrogram(Y_scale, sr, HOP_SIZE)

            #Log_Amplitude Spectrogram

            Y_log_scale = librosa.power_to_db(Y_scale)

            plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis = 'log')