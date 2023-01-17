#This project is based on audio speech text recognitoion in k 
#use the link for more detail on mfcc http://recognize-speech.com/feature-extraction/mfcc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#dataset link https://www.kaggle.com/sushantjha8/audio-words
#I would like to thank 
#I would like to thank my friend who appreciated me to work on this project.
#My prof. Dr. Yaduvir Singh 
#My friend/colleague : Santosh Srivastav
#refrence link :- https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b
#furthure research on CNN + RNN for series prediction :
# https://wiki.tum.de/display/lfdv/Recurrent+Neural+Networks+-+Combination+of+RNN+and+CNN
#refrence for CNN+LSTM in keras
#https://github.com/keras-team/keras/issues/5527
import os
import librosa
import os
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical


DATA_PATH = "../"

 #Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

# convert wav2mfcc
def mfcc1(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = mfcc1(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state)

def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data

def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]
save_data_to_array(path=DATA_PATH, max_pad_len=11)


# print(prepare_dataset(DATA_PATH))
X_train, X_test, y_train, y_test = get_train_test()

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1) #reshape for single length size
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
print(X_test.shape)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
"""Model designing of speech to text """
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))#give dropout for each itration for over fitting
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))#give dropout for each itration for over fitting
model.add(Dense(7, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(X_train, y_train_hot, batch_size=100, epochs=200, verbose=1, validation_data=(X_test, y_test_hot))

""" This model provide 0.908 acc."""
"""Predicting Word"""
filefo=input("Enter wav file address       ")
sample = mfcc1(filefo)

# We need to reshape it remember?
sample_reshaped = sample.reshape(1, 20, 11, 1)

# Perform forward pass
print(get_labels()[0][
    np.argmax(model.predict(sample_reshaped))
])