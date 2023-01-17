import pandas as pd

import glob, os ,random

import numpy as np

import librosa, librosa.display

import matplotlib.pyplot as plt
base_path = "../input/birdsong-recognition/train_audio"
import warnings

warnings.filterwarnings('ignore')
R_count = 264

C_count = 4     

SUB_count = 1 

FIG_SIZE = (20,600)



#loop through the base directory 

for i, direc in enumerate(os.listdir(base_path)):

    

    #choose a random audio from each direc

    file_name  = random.choice(os.listdir('{}/{}'.format(base_path,direc)))  

    file = os.path.join('{}/{}/{}'.format(base_path,direc,file_name))



    

    # load audio file with Librosa

    signal, sample_rate = librosa.load(file)

    fft = np.fft.fft(signal)



    # calculate abs values on complex numbers to get magnitude

    spectrum = np.abs(fft)



    # create frequency variable

    f = np.linspace(0, sample_rate, len(spectrum))



    # need only one side ,take half of the spectrum and frequency

    left_spectrum = spectrum[:int(len(spectrum)/2)]

    left_f = f[:int(len(spectrum)/2)]

    

    hop_length = 512 # number of samples between each successive FFT window

    n_fft = 2048 # number of samples in a single window



    # calculate duration hop length and window in seconds

    hop_length_duration = float(hop_length)/sample_rate

    n_fft_duration = float(n_fft)/sample_rate

    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)



    # calculate abs values to get magnitude

    spectrogram = np.abs(stft)

    

    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    



    fig = plt.figure(figsize=FIG_SIZE)



    fig.add_subplot(R_count,C_count, 1)

    librosa.display.waveplot(signal, sample_rate, alpha=0.4)

    plt.xlabel("Time (s)")

    plt.ylabel("Amplitude")

    plt.title('{}.Waveform-{}'.format(i+1,direc.upper()))







    fig.add_subplot(R_count, C_count, 2)

    plt.plot(left_f, left_spectrum, alpha=0.4)

    plt.xlabel("Frequency")

    plt.ylabel("Magnitude")

    plt.title('{}.Power spectrum-{}'.format(i+1,direc.upper()))







    fig.add_subplot(R_count,C_count, 3)

    librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)

    plt.xlabel("Time")

    plt.ylabel("Frequency")

    plt.colorbar(format="%+2.0f dB")

    plt.title('{}.Spectrogram (dB)-{}'.format(i+1,direc.upper()))



    

    fig.add_subplot(R_count, C_count, 4)

    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)

    plt.xlabel("Time")

    plt.ylabel("MFCC coefficients")

    plt.colorbar()

    plt.title('{}.MFCCs-{}'.format(i+1,direc.upper()))



    

    plt.tight_layout()

    plt.show()    