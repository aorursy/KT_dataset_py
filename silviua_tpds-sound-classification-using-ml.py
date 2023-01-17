# Data libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import scipy.io.wavfile as sci_wav  # Open wav files

import os  # Manipulate files

import warnings  # Warning removal

import glob

import ntpath

import librosa

warnings.filterwarnings('ignore')



# List the wav files

ROOT_DIR_TEST = glob.glob('../input/audio-cats-and-dogs/cats_dogs/test')[0]

ROOT_DIR_TRAIN = glob.glob('../input/audio-cats-and-dogs/cats_dogs/train')[0]





X_path = glob.glob(ROOT_DIR_TEST + "/test/*") # test = dogs in this case ! (wrong name of directory was given when it was created)

X_path = X_path + glob.glob(ROOT_DIR_TEST + "/cats/*")

X_path = X_path + glob.glob(ROOT_DIR_TRAIN + "/dog/*")

X_path = X_path + glob.glob(ROOT_DIR_TRAIN + "/cat/*")

print (len(X_path))

y = np.empty((0, 1, ))

for f in X_path:

    if 'cat' in ntpath.basename(f):

        resp = np.array([0])

        resp = resp.reshape(1, 1, )

        y = np.vstack((y, resp))

    elif 'dog' in ntpath.basename(f):

        resp = np.array([1])

        resp = resp.reshape(1, 1, )

        y = np.vstack((y, resp))



# Split train and test

X_train, X_test, y_train, y_test = train_test_split(X_path, y, test_size=0.33)
print("in X_train, there is {} cats and {} dogs".format(len(y_train) - sum(y_train), sum(y_train)))

print("in X_test, there is {} cats and {} dogs".format(len(y_test) - sum(y_test), sum(y_test)))



print (y_test.shape)
def read_wav_files(wav_files):

    '''Returns a list of audio waves

    Params:

        wav_files: List of .wav paths

    

    Returns:

        List of audio signals

    '''

    if not isinstance(wav_files, list):

        wav_files = [wav_files]

    return [sci_wav.read(f)[1] for f in wav_files]
def librosa_read_wav_files(wav_files):

    if not isinstance(wav_files, list):

        wav_files = [wav_files]

    return [librosa.load(f)[0] for f in wav_files]
wav_rate = librosa.load(X_train[0])[1]

X_train = librosa_read_wav_files(X_train)

X_test  = librosa_read_wav_files(X_test)

import matplotlib.pyplot as plt



fig, axs = plt.subplots(2, 2, figsize=(16,7))

axs[0][0].plot(X_train[0])

axs[0][1].plot(X_train[1])

axs[1][0].plot(X_train[2])

axs[1][1].plot(X_train[3])

plt.show()
import IPython.display as ipd

# ipd.Audio('../input/audio-cats-and-dogs/cats_dogs/dog_barking_27.wav')

ipd.Audio(X_train[0],  rate=wav_rate)

def zero_crossing_rate(wavedata, block_length, sample_rate):

    

    # how many blocks have to be processed?

    num_blocks = int(np.ceil(len(wavedata)/block_length))

    

    # when do these blocks begin (time in seconds)?

    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(sample_rate)))

    

    zcr = []

    

    for i in range(0,num_blocks-1):

        

        start = i * block_length

        stop  = np.min([(start + block_length - 1), len(wavedata)])

        

        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))

        zcr.append(zc)

    

    return np.asarray(zcr), np.asarray(timestamps)


ifeatures, itimestamp = zero_crossing_rate(X_train[0], int(len(X_train[0])/10), wav_rate)



print (ifeatures)

print (itimestamp)
plt.plot(itimestamp, ifeatures)
zero_cross_feat = librosa.feature.zero_crossing_rate(X_train[0])



print (zero_cross_feat.mean())

plt.plot(zero_cross_feat[0])
# function to extract all the features needed for the classification

def extract_features(audio_samples, sample_rate):

    extracted_features = np.empty((0, 41, ))

    if not isinstance(audio_samples, list):

        audio_samples = [audio_samples]

        

    for sample in audio_samples:

        # calculate the zero-crossing feature

        zero_cross_feat = librosa.feature.zero_crossing_rate(sample).mean()

        

        # calculate the mfccs features

        mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=40)

        mfccsscaled = np.mean(mfccs.T,axis=0)



        # add zero crossing feature to the feature list

        mfccsscaled = np.append(mfccsscaled, zero_cross_feat)

        mfccsscaled = mfccsscaled.reshape(1, 41, )

        

        extracted_features = np.vstack((extracted_features, mfccsscaled))



    # return the extracted features

    return extracted_features
features = ((extract_features(X_train[0], wav_rate)))



print (len(features))

print (features.shape)

plt.plot(features[0])
X_train_features = extract_features(X_train, wav_rate)

X_test_features  = extract_features(X_test, wav_rate)
print("Image array shape: ", X_train_features.shape)

print("Image array shape: ", X_test_features.shape)

print("Label array shape: ", y_train.shape)

print("Label array shape: ", y_test.shape)
from keras import layers

from keras import models

from keras import optimizers

from keras import losses

from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical

train_labels = to_categorical(y_train)

test_labels = to_categorical(y_test)

model = models.Sequential()



model.add(layers.Dense(100, activation = 'relu', input_shape = (41, )))

model.add(layers.Dense(50, activation = 'relu'))

model.add(layers.Dense(2, activation = 'softmax'))



model.summary()
best_model_weights = './base.model'

checkpoint = ModelCheckpoint(

    best_model_weights,

    monitor='val_acc',

    verbose=1,

    save_best_only=True,

    mode='min',

    save_weights_only=False,

    period=1

)



callbacks = [checkpoint]



model.compile(optimizer='adam',

              loss=losses.categorical_crossentropy,

              metrics=['accuracy'])


history = model.fit(

    X_train_features,

    train_labels,

    validation_data=(X_test_features,test_labels),

    epochs = 300, 

    verbose = 1,

    callbacks=callbacks,

)
#Save the model

model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')

...

# list all data in history

print(history.history.keys())



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'b', label = "training accuracy")

plt.plot(epochs, val_acc, 'r', label = "validation accuracy")

plt.title('Training and validation accuracy')

plt.legend()



plt.show()
nr_to_predict = 24

pred = model.predict(X_test_features[nr_to_predict].reshape(1, 41,))



print("Cat: {} Dog: {}".format(pred[0][0], pred[0][1]))



if (y_test[nr_to_predict] == 0):

    print ("The label says that it is a Cat!")

else:

    print ("The label says that it is a Dog!")

    

plt.plot(X_test_features[nr_to_predict])

ipd.Audio(X_test[nr_to_predict],  rate=wav_rate)