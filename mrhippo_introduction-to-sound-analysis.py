# data analysis and visualization
import numpy as np 
import pandas as pd 
import librosa 
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
import IPython.display as ipd
from scipy.io import wavfile
import glob

# keras and scikitlearn/deep learning
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense, Dropout, Input, ReLU, LeakyReLU, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def sine_wave(amp, freq, time, phase):
    
    y_list = []
    for t in (time):
        y = amp * np.sin(2 * np.pi * freq * t + phase)
        y_list.append(y)
        
    return np.array(y_list)

wave = sine_wave(3, .5, np.arange(0,6, 0.0001), 0)

plt.figure(figsize = (15,8))
plt.plot(wave, label = "Sine Wave")
plt.legend()
plt.show()
def cos_wave(amp, freq, time, phase):
    
    y_list = []
    for t in (time):
        y = amp * np.cos(2 * np.pi * freq * t + phase)
        y_list.append(y)
        
    return np.array(y_list)

s_wave = sine_wave(3, 0.5, np.arange(0,6, 0.0001), 0)
c_wave = cos_wave(3, 0.5, np.arange(0,6, 0.0001), 0) 

plt.figure(figsize = (15,8))
plt.plot(s_wave, label = "Sine Wave")
plt.plot(c_wave, label = "Cosine Wave")
plt.text(100, c_wave[0], "Cos", fontsize=15, color = "orange")
plt.text(100, s_wave[0], "Sin", fontsize=15, color = "blue")
plt.legend()
plt.show()
def pitch_to_frequency(p):
    
    freqs = []
    
    for i in p:
        f = 2**(((i-69)/12))*440
        freqs.append(f)
    
    return freqs

pitches = np.arange(21, 105, 0.001) # 21, 33, 45, 57, 69, 81, 93, 105
freqs = pitch_to_frequency(pitches)

plt.figure(figsize = (15,8))
plt.plot(freqs, pitches)
plt.grid(alpha = 0.6)
plt.ylabel("Pitch")
plt.xlabel("Frequency")
plt.title("Mapping Pitch to Frequency")
plt.show()
s1 = sine_wave(2,0.8, np.arange(0,6,0.0001), 0)
s2 = sine_wave(2,2, np.arange(0,6,0.0001), 0)
s3 = sine_wave(2,4, np.arange(0,6,0.0001), 0)

s1, s2, s3 = np.array(s1), np.array(s2), np.array(s3) 

plt.figure(figsize = (15,8))

ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1)
ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=1)
ax3 = plt.subplot2grid((2, 3), (0, 2), colspan=1)
ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)

ax4.plot(s1 + s2 + s3, color = "red")
ax4.set_title("Waves Added (Complex wave)")

ax1.plot(s1)
ax1.set_title("Wave 1")

ax2.plot(s2, color = "orange")
ax2.set_title("Wave 2")

ax3.plot(s3, color = "green")
ax3.set_title("Wave 3")

plt.show()
s1 = sine_wave(2,0.8, np.arange(0,6,0.0001), 0)
s2 = sine_wave(2,0.8, np.arange(0,6,0.0001), 0)

s2 = np.array(s2)
 
reverser = np.full(
    shape=len(s2),
    fill_value=-1,
    dtype=np.int)

s2 = s2*reverser

plt.figure(figsize = (15,6))
plt.plot(s1, alpha = 0.4)
plt.plot(s2, alpha = 0.4)
plt.plot(s2 + s1, color = "black")
plt.title("Canceling Waves")
plt.show()
df = pd.read_csv("../input/freesound-audio-tagging/train.csv")
df.head()
df_saxophone = df[df["label"] == "Saxophone"]
df_saxophone.head()
files = []

for path in df_saxophone["fname"]: # we will use saxophone samples
    file, _ = librosa.core.load("../input/freesound-audio-tagging/audio_train/" + str(path), sr=44100) # 44100Hz in dataset
    files.append(file)
sample = files[3] # the smaple file we will use
IPython.display.display(ipd.Audio(data = sample, rate = 44100)) # the original sample rate of recording
sample_rates = [60000, 22050, 16000, 10000]

for sr in sample_rates:
    IPython.display.display(ipd.Audio(data = sample, rate = sr))
plt.figure(figsize = (15, 8))
plt.plot(sample)
plt.show()
plt.figure(figsize = (20, 8))
plt.plot(sample[:500])
plt.plot(sample[:500], ".", color = "red")
plt.show()
fft = np.fft.fft(sample)

magnitude = np.abs(fft)
freq = np.linspace(0, 44100, len(magnitude))

plt.figure(figsize = (15, 8))
plt.plot(freq, magnitude)
plt.title("Fast Fourier Transform")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
left_freq = freq[:int(len(freq)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

plt.figure(figsize = (15, 8))
plt.plot(left_freq, left_magnitude)
plt.title("Fixed Fast Fourier Transform")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
fig, ax = plt.subplots(3,1, figsize = (18,12))

ax[0].plot(left_freq[:500], left_magnitude[:500])
ax[0].set_title("FFT Zoom: 0-500")
ax[0].set_ylabel("Magnitude")

ax[1].plot(left_freq[1000:1500], left_magnitude[1000:1500])
ax[1].set_title("FFT Zoom: 1000-1500")
ax[1].set_ylabel("Magnitude")

ax[2].plot(left_freq[5000:5500], left_magnitude[5000:5500])
ax[2].set_title("FFT Zoom: 5000-5500")
ax[2].set_xlabel("Frequency")
ax[2].set_ylabel("Magnitude")

plt.show()
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(sample, hop_length = hop_length, n_fft = n_fft)
spectrogram = np.abs(stft)

plt.figure(figsize = (15,8))
librosa.display.specshow(spectrogram, sr = 44100, hop_length = hop_length)
plt.title("Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()
log_spectrogram = librosa.amplitude_to_db(spectrogram) # applying logarithm

plt.figure(figsize = (15,8))
librosa.display.specshow(log_spectrogram, sr = 44100, hop_length = hop_length)
plt.title("Log Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()
MFCCs = librosa.feature.mfcc(sample, n_fft = n_fft, hop_length = hop_length, n_mfcc = 13)

plt.figure(figsize = (15,8))
librosa.display.specshow(MFCCs, sr = 44100, hop_length = hop_length)
plt.title("MFCCs")
plt.xlabel("Time")
plt.ylabel("MFCC Coefficents")
plt.colorbar()
plt.show()
print(df.label.unique())
print("Number of Instruments: ", len(df.label.unique()))
DURATION = 4
SAMPLE_RATE = 16000
AUDIO_SHAPE = SAMPLE_RATE*DURATION

# Path
DATASET_PATH = "../input/freesound-audio-tagging/"

# Load 
def load_train_data(label, input_length=AUDIO_SHAPE):
    train = pd.read_csv(DATASET_PATH + "train.csv")
    if label != None:
        train_list = train.loc[train.label == label]
    else: 
        train_list = train
    cur_batch_size = len(train_list)
    train_fname_list = train_list.fname
    X = np.empty((cur_batch_size, input_length))
    for i, train_fname in enumerate(train_fname_list):
        file_path = DATASET_PATH + "audio_train/" + train_fname
        
        # Read and Resample the audio
        data, _ = librosa.core.load(file_path, sr=SAMPLE_RATE, res_type='kaiser_fast')

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        X[i,] = data
    print("{} Files Loaded".format(label))
    return X

def normalization(X):
    mean = X.mean(keepdims=True)
    std = X.std(keepdims=True)
    X = (X - mean) / std
    return X

# Rescale Data to be in range [rangeMin, rangeMax]
def rescale(X, rangeMin=-1, rangeMax=+1):
    maxi = X.max()
    mini = X.min()
    X = np.interp(X, (mini, maxi), (rangeMin, rangeMax))
    return X
instruments = ["Hi-hat", "Trumpet", "Cello", "Clarinet", "Snare_drum", "Flute", "Bass_drum", "Electric_piano", "Acoustic_guitar", "Violin_or_fiddle"]

X_train = []
Y_train = []

for label, instrument in enumerate(instruments, start = 0):
    
    x = load_train_data(instrument)
    x = normalization(x)
    x = rescale(x)
            
    for file in x:
        
        X_train.append(file)
        Y_train.append(label)
print(len(X_train))
print(len(Y_train))
X_train = np.array(X_train)
Y_train = np.array(Y_train)
fig = plt.figure(figsize = (12,8))
sns.countplot(Y_train,palette = "Blues")
plt.title("Label Count")
plt.show()
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)
y_train = to_categorical(y_train,num_classes = 10)
y_test = to_categorical(y_test,num_classes = 10)
print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)
print("x_test shape: ",x_test.shape)
print("y_test shape: ",y_test.shape)
model = Sequential()

model.add(Dense(512, kernel_regularizer = keras.regularizers.l2(0.001), input_shape = (x_train.shape[0], x_train.shape[1])))
model.add(ReLU())
model.add(Dropout(0.4))

model.add(Dense(256, kernel_regularizer = keras.regularizers.l2(0.001)))
model.add(ReLU())
model.add(Dropout(0.3))

model.add(Dense(64, kernel_regularizer = keras.regularizers.l2(0.001)))
model.add(ReLU())
model.add(Dropout(0.2))

model.add(Dense(10, activation = "softmax"))

model.compile(loss = "categorical_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"])

history = model.fit(x_train,y_train,epochs = 20, batch_size = 100,validation_data = (x_test,y_test))
history.history.keys()
plt.figure(figsize = (13,8))
plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"],label = "Validation Accuracy")
plt.title("Accuracies")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.figure(figsize = (13,8))
plt.plot(history.history["loss"], label = "Train Loss")
plt.plot(history.history["val_loss"],label = "Validation Loss")
plt.title("Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
mfcc_train = []

for file in X_train:
    MFCCs = librosa.feature.mfcc(file, n_fft = n_fft, hop_length = hop_length, n_mfcc = 15)
    mfcc_train.append(MFCCs.T) 
mfcc_train = np.array(mfcc_train)
mfcc_train.shape
x_train, x_test, y_train, y_test = train_test_split(mfcc_train, Y_train, test_size = 0.2, random_state = 42)
y_train = to_categorical(y_train,num_classes = 10)
y_test = to_categorical(y_test,num_classes = 10)
print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)
print("x_test shape: ",x_test.shape)
print("y_test shape: ",y_test.shape)
x_train = x_train.reshape(x_train.shape[0], 126, 15, 1)
x_test = x_test.reshape(x_test.shape[0], 126, 15, 1)
print(x_train.shape)
print(x_test.shape)
model = Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(keras.layers.MaxPool2D((3, 3), strides = (2, 2), padding = "same"))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPool2D((3, 3), strides = (2, 2), padding = "same"))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPool2D((3, 3), strides = (2, 2), padding = "same"))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.3))

model.add(Dense(10, activation = "softmax"))

model.compile(loss = "categorical_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"])

history = model.fit(x_train,y_train,epochs = 30, batch_size = 100,validation_data = (x_test,y_test))
history.history.keys()
plt.figure(figsize = (13,8))
plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"],label = "Validation Accuracy")
plt.title("Accuracies")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.figure(figsize = (13,8))
plt.plot(history.history["loss"], label = "Train Loss")
plt.plot(history.history["val_loss"],label = "Validation Loss")
plt.title("Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()