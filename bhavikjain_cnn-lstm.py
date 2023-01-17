import os

from glob import glob

import pickle

import itertools

import numpy as np

from scipy.stats import zscore

from sklearn.model_selection import train_test_split



### Graph imports ###

import matplotlib.pyplot as plt

from PIL import Image

import pandas as pd



### Audio import ###

import librosa

import IPython

from IPython.display import Audio



### Plot imports ###

from IPython.display import Image

import matplotlib.pyplot as plt



### Time Distributed ConvNet imports ###

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed, concatenate

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, LeakyReLU, Flatten

from tensorflow.keras.layers import LSTM

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import backend as K

from keras.utils import np_utils

from keras.utils import plot_model

from sklearn.preprocessing import LabelEncoder



### Warning ###

import warnings

warnings.filterwarnings('ignore')
# Audio file path and names

RAV = '/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/'
dic  =  {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}
dir_list = os.listdir(RAV)

dir_list.sort()



emotion = []

path = []

for i in dir_list:

    fname = os.listdir(RAV + i)

    for f in fname:

        part = f.split('.')[0].split('-')

        emotion.append(int(part[2]))

        path.append(RAV + i + '/' + f)



RAV_df = pd.DataFrame() 

RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)
RAV_df.head()
signal = []



# Sample rate (16.0 kHz)

sample_rate = 16000     



# Max pad lenght (3.0 sec)

max_pad_len = 49100



for index,path in enumerate(RAV_df.path):

    X, sample_rate = librosa.load(path

                                  ,duration=3

                                  ,offset=0.5

                                 )

    sample_rate = np.array(sample_rate)

    

    y = zscore(X)

        

    # Padding or truncated signal 

    if len(y) < max_pad_len:    

        y_padded = np.zeros(max_pad_len)

        y_padded[:len(y)] = y

        y = y_padded

    elif len(y) > max_pad_len:

        y = np.asarray(y[:max_pad_len])



    # Add to signal list

    signal.append(y)

    
labels = np.asarray(emotion).ravel()
labels.shape
np.unique(labels)
def noisy_signal(signal, snr_low=15, snr_high=30, nb_augmented=2):

    

    # Signal length

    signal_len = len(signal)



    # Generate White noise

    noise = np.random.normal(size=(nb_augmented, signal_len))

    

    # Compute signal and noise power

    s_power = np.sum((signal / (2.0 ** 15)) ** 2) / signal_len

    n_power = np.sum((noise / (2.0 ** 15)) ** 2, axis=1) / signal_len

    

    # Random SNR: Uniform [15, 30]

    snr = np.random.randint(snr_low, snr_high)

    

    # Compute K coeff for each noise

    K = np.sqrt((s_power / n_power) * 10 ** (- snr / 10))

    K = np.ones((signal_len, nb_augmented)) * K

    

    # Generate noisy signal

    return signal + K.T * noise
print("Data Augmentation: START")

augmented_signal = list(map(noisy_signal, signal))

print("Data Augmentation: END!")
def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):

    

    # Compute spectogram

    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

    

    # Compute mel spectrogram

    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)

    

    # Compute log-mel spectrogram

    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    

    return mel_spect
mel_spect = np.asarray(list(map(mel_spectrogram, signal)))

augmented_mel_spect = [np.asarray(list(map(mel_spectrogram, augmented_signal[i]))) for i in range(len(augmented_signal))]
MEL_SPECT_train, MEL_SPECT_test, AUG_MEL_SPECT_train, AUG_MEL_SPECT_test, label_train, label_test = train_test_split(mel_spect, augmented_mel_spect,labels, test_size=0.2,random_state=1)



# Build augmented labels and train

aug_label_train = np.asarray(list(itertools.chain.from_iterable([[label] * 2 for label in label_train])))

AUG_MEL_SPECT_train = np.asarray(list(itertools.chain.from_iterable(AUG_MEL_SPECT_train)))



# Concatenate original and augmented

X_train = np.concatenate((MEL_SPECT_train, AUG_MEL_SPECT_train))

y_train = np.concatenate((label_train, aug_label_train))



# Build test set

X_test = MEL_SPECT_test

y_test = label_test
X_train.shape
y_train
win_ts = 128

hop_ts = 64



# Split spectrogram into frames

def frame(x, win_step=128, win_size=64):

    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)

    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)

    for t in range(nb_frames):

        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)

    return frames



# Frame for TimeDistributed model

X_train = frame(X_train, hop_ts, win_ts)

X_test = frame(X_test, hop_ts, win_ts)
lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(np.ravel(y_train)))

y_test = np_utils.to_categorical(lb.transform(np.ravel(y_test)))
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , X_train.shape[2], X_train.shape[3], 1)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , X_test.shape[2], X_test.shape[3], 1)
K.clear_session()



# Define two sets of inputs: MFCC and FBANK

input_y = Input(shape=X_train.shape[1:], name='Input_MELSPECT')



## First LFLB (local feature learning block)

y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT')(input_y)

y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)

y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)

y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT')(y)

y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)     



## Second LFLB (local feature learning block)

y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT')(y)

y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)

y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)

y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT')(y)

y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)



## Second LFLB (local feature learning block)

y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT')(y)

y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)

y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)

y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT')(y)

y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)



## Second LFLB (local feature learning block)

y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT')(y)

y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)

y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)

y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT')(y)

y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)  



## Flat

y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)                      

                               

# Apply 2 LSTM layer and one FC

y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)

y = Dense(y_train.shape[1], activation='softmax', name='FC')(y)



# Build final model

model = Model(inputs=input_y, outputs=y)

model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.8), loss='categorical_crossentropy', metrics=['accuracy'])



# Early stopping

early_stopping = EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='max')



# Fit model

history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])
score = model.evaluate(X_test, y_test, verbose=0)

score
model.save('[CNN-LSTM]M.h5')

model.save_weights('[CNN-LSTM]W.h5')
model_name = 'Emotion_Model.h5'

save_dir = os.path.join(os.getcwd(), 'saved_models')



if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Save model and weights at %s ' % model_path)



# Save the model to disk

model_json = model.to_json()

with open("model_json.json", "w") as json_file:

    json_file.write(model_json)
preds = model.predict(X_test)



preds=preds.argmax(axis=1)

preds
path_ = '/kaggle/input/bhahahahhhahahh/download.wav'
import IPython.display as ipd

ipd.Audio(path_)
s = []



# Sample rate (16.0 kHz)

sample_rate = 16000     



# Max pad lenght (3.0 sec)

max_pad_len = 49100



X, sample_rate = librosa.load(path_,duration=3,offset=0.5)

sample_rate = np.array(sample_rate)



y = zscore(X)

    

# Padding or truncated signal 

if len(y) < max_pad_len:    

    y_padded = np.zeros(max_pad_len)

    y_padded[:len(y)] = y

    y = y_padded

elif len(y) > max_pad_len:

    y = np.asarray(y[:max_pad_len])



# Add to signal list

s.append(y)
mel_spect = np.asarray(list(map(mel_spectrogram, s)))
win_ts = 128

hop_ts = 64



# Split spectrogram into frames

def frame(x, win_step=128, win_size=64):

    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)

    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)

    for t in range(nb_frames):

        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)

    return frames



# Frame for TimeDistributed model

x = frame(mel_spect, hop_ts, win_ts)
x = x.reshape(x.shape[0], x.shape[1] , x.shape[2], x.shape[3], 1)
preds = model.predict(x)
preds=preds.argmax(axis=1)

preds