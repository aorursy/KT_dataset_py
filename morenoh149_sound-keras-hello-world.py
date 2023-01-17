import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
# install in kaggle kernel
# if your sound files are .wav, scipy.io.wavfile is more reliable
# this module sometimes prevents us from using kaggle GPU's
import soundfile as sf
train_dir = '../input/songs/songs'
df = pd.read_csv('../input/birdsong_metadata.csv')
df.head()
# find NaNs in metadata
df[df.isnull().any(axis=1)]
# num of samples
len(df)
df[['genus', 'species']].describe()
# load sounds from disk
# takes 1 minute
paths = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
dataset = []
for p in tqdm(paths):
    audio, _ = sf.read(p)
    dataset.append(audio)
dataset
# Find length of shortest sound, used later to normalize sound lengths
# TODO pad with https://keras.io/preprocessing/sequence/#pad_sequences
min_length = min([len(x) for x in dataset])
min_length
# clip all samples to the length of the shortest sample so they fit into keras
for i, x in enumerate(dataset):
    dataset[i] = x[:min_length]
# convert list of numpy arrays into single numpy array for consumption
x_train = np.array(dataset)
x_train.shape
# naive label object to get keras model to compile.
# TODO convert np.array of sample labels into one-hot-vector-label
y_train = keras.utils.to_categorical([0, 1, 2], num_classes=3)
y_train
# Define model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=x_train[0].shape))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train model
model.fit(x_train, y_train, epochs=10, batch_size=32)
# TODO use model to predict category on test set