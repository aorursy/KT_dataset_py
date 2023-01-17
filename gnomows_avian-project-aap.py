""" `imageio_ffmpeg` contains a pre-built `ffmpeg` binary, needed for mp3 decoding by `librosa`. 

    It is installed as a custom package on Kaggle. If no `ffmpeg` binary is found in 

    `/usr/local/bin` then create a softlink to the `imageio_ffmpeg` binary. 

"""

!pip install imageio_ffmpeg



import os

if not os.path.exists("/usr/local/bin/ffmpeg"): 

    #! pip install imageio_ffmpeg 

    import imageio_ffmpeg

    os.link(imageio_ffmpeg.get_ffmpeg_exe(), "/usr/local/bin/ffmpeg")
""" Common imports """

from matplotlib.gridspec import GridSpec

import keras

from keras.callbacks import ModelCheckpoint

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential

from keras.utils import to_categorical

import re

from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt

from glob import glob

import pandas as pd

import numpy as np





data_dir = '../input/xenocanto-avian-vocalizations-canv-usa/xeno-canto-ca-nv/'

df = pd.read_csv("../input/xenocanto-avian-vocalizations-canv-usa/xeno-canto_ca-nv_index.csv").drop('Unnamed: 0',axis=1)



label_encoder = LabelEncoder().fit(df['english_cname'] )

n_classes = len(label_encoder.classes_)

print("The dataset contains %i distinct species labels."%n_classes)

print("%i mp3s found in %s"%(len(glob(data_dir+"*.mp3")), data_dir))



y_encoded_entire_dataset = np.array(label_encoder.transform(df['english_cname']))



plt.figure(figsize=(15,2))

plt.title("Distribution of Samples per Species")

plt.hist(y_encoded_entire_dataset, bins=n_classes)

plt.xlim(-1,91)

plt.ylabel("Number of Samples")

plt.xticks(range(n_classes), label_encoder.classes_, rotation='vertical')

plt.show()
label_encoder = LabelEncoder().fit(df['duration_seconds'] )

n_classes = len(label_encoder.classes_)

y_encoded_entire_dataset = np.array(label_encoder.transform(df['duration_seconds']))



plt.figure(figsize=(15,2))

plt.title("Distribution of Time-lenght in all samples")

plt.hist(y_encoded_entire_dataset, bins=n_classes)

plt.xlim(-1,91)

plt.ylabel("Number of Samples")

plt.xlabel("Length of audio")

plt.xticks(range(n_classes), label_encoder.classes_, rotation='vertical')

plt.show()
from librosa.display import specshow

from IPython.display import Audio

from itertools import islice

from scipy import signal

import librosa



Abert = df[df.english_cname.str.contains('Abert\'s Towhee')]  #  Filtering over english name (for example)



hop_length = 512

n_fft = 2048



plt.rcParams["figure.figsize"] = (16,6)

for i, sample in islice(Abert.iterrows(), 0, Abert.shape[0]):

    print("%s: %s, contributed by: %s %s"%(sample.file_name, sample.full_name, sample.recordist, sample.recordist_url))

    data, samplerate = librosa.load("../input/xenocanto-avian-vocalizations-canv-usa/xeno-canto-ca-nv/" + sample.file_name)

    display(Audio(data, rate=samplerate))

    

    # Trim data and show

    data_t, _ = librosa.effects.trim(data)

    librosa.display.waveplot(data_t, sr=samplerate)

    plt.title("Time domain")

    plt.show()

    

    # STFT of normal data

    D = np.abs(librosa.stft(data_t[:n_fft], n_fft=n_fft, hop_length=n_fft+1))

    plt.plot(D)

    plt.title("Short-time frequency domain")

    plt.show()

    

    # Normal melspec

    sg = librosa.feature.melspectrogram(data, sr=samplerate, hop_length=hop_length, n_fft=n_fft)

    

    # Normal stft

    X = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)

    

    # Creation of the filter

    cutOff = 1000 # Cutoff frequency

    N  = 6    # Filter order

    nyq = 0.5 * samplerate

    fc = cutOff / nyq # Cutoff frequency normal

    b, a = signal.butter(N, fc)



    # Apply the filter over data

    tempf = signal.filtfilt(b, a, data)

    

#     D1 = np.abs(librosa.stft(tempf[:n_fft], n_fft=n_fft, hop_length=n_fft+1))

#     plt.plot(D1)



    # Filtered STFT

    X_after_filter = librosa.stft(tempf, n_fft=n_fft, hop_length=hop_length)

    

    # Filtered_melspec

    f_sg = librosa.feature.melspectrogram(tempf, sr=samplerate, hop_length=hop_length, n_fft=n_fft)



    fig, axs = plt.subplots(2, 2)

    fig.suptitle("%s: %s"%(sample.file_name, sample.full_name))

    

    specshow(np.log(sg), y_axis='mel', x_axis='time', ax=axs[0][0], hop_length=hop_length)

    axs[0][0].set_title("log(Melspectrogram)")

    

    axs[1][0].hist(np.log(sg.flatten()), bins=100)

    axs[1][0].set_title("Histogram of log(Melspectrogram)")

    

    # 

    librosa.display.specshow(X, sr=samplerate, hop_length=hop_length, ax=axs[0][1], x_axis='time', y_axis='linear')

    axs[0][1].set_title("SFTF")

    

#     specshow(np.log(f_sg), y_axis='mel', x_axis='time', ax=axs[1][1], hop_length=hop_length)

    librosa.display.specshow(X_after_filter, sr=samplerate, hop_length=hop_length, ax=axs[1][1], x_axis='time', y_axis='linear')

    axs[1][1].set_title("Histogram of filtered log(Melspectrogram)")

    

    plt.show()
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from sklearn.preprocessing import LabelEncoder



y_english_labels_entire_dataset = [s['english_cname'] for i,s in df.iterrows()]

label_encoder = LabelEncoder().fit(y_english_labels_entire_dataset)

y_encoded_entire_dataset = np.array(label_encoder.transform(y_english_labels_entire_dataset))



n_classes = len(label_encoder.classes_)



X_train, X_test, y_train, y_test = train_test_split(

    np.array([s['file_id'] for i,s in df.iterrows()]), 

    y_encoded_entire_dataset, 

    test_size=1/5, 

    stratify=y_encoded_entire_dataset, 

    shuffle=True, 

    random_state=37,

)

print("Training data shape:",X_train.shape, y_train.shape)

print("Test data shape:    ",X_test.shape, y_test.shape)
Abert = df[df.english_cname.str.contains('Abert\'s Towhee')] # Other type of filter 



hop_length = 512

n_fft = 2048



plt.rcParams["figure.figsize"] = (16,6)

for i, sample in islice(Abert.iterrows(), 0, 2):

    print("%s: %s, contributed by: %s %s"%(sample.file_name, sample.full_name, sample.recordist, sample.recordist_url))

    data, samplerate = librosa.load("../input/xenocanto-avian-vocalizations-canv-usa/xeno-canto-ca-nv/" + sample.file_name)

    display(Audio(data, rate=samplerate))

    

    # Trim data and show

    data_t, _ = librosa.effects.trim(data)

    librosa.display.waveplot(data_t, sr=samplerate)

    plt.title("Time domain")

    plt.show()



    # Creation of the filter

    cutOff = 1000 # Cutoff frequency

    N  = 6    # Filter order

    nyq = 0.5 * samplerate

    fc = cutOff / nyq # Cutoff frequency normal

    b, a = signal.butter(N, fc)



    # Apply the filter over data

    tempf = signal.filtfilt(b, a, data)

    

    D1 = np.abs(librosa.stft(tempf[:n_fft], n_fft=n_fft, hop_length=n_fft+1))

    plt.plot(D1)

    plt.show()

    

    

    display(Audio(tempf, rate=samplerate))

    

    # Trim data and show

    data_t, _ = librosa.effects.trim(tempf)

    librosa.display.waveplot(data_t, sr=samplerate)

    plt.title("Time domain")

    plt.show()