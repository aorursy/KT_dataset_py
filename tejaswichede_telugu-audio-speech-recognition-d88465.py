# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import libraries 

import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import glob 

from sklearn.metrics import confusion_matrix

import IPython.display as ipd  # To play sound in the notebook

import os

import sys

import warnings

# ignore warnings 

if not sys.warnoptions:

    warnings.simplefilter("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning) 
#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



#TESS = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"

#RAV = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

SAVEE = "../input/speechdataset/my-Audio-Dataset/Emotions/Angry/"

#CREMA = "/kaggle/input/cremad/AudioWAV/"

# Run one example 

dir_list = os.listdir(SAVEE)

dir_list[0:16]
# Get the data location for SAVEE

dir_list = os.listdir(SAVEE)



# parse the filename to get the emotions

emotion=[]

path = []

for i in dir_list:

    if i[-20:-19]=='A':

        emotion.append('male_angry')

   

    elif i[-20:-19]=='A':

        emotion.append('Female_Angry')

   

   #elif i[-8:-6]=='sa':

   #     emotion.append('male_sad')

    else:

        emotion.append('male_error') 

    path.append(SAVEE + i)

# Now check out the label count distribution 

SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])

SAVEE_df['source'] = 'SAVEE'

SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)

SAVEE_df.labels.value_counts()
# use the well known Librosa library for this task 

fname = SAVEE + 'Angry.01.male.01.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
# Import our libraries

import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import specgram

import pandas as pd

import os

import IPython.display as ipd  # To play sound in the notebook
# Source - RAVDESS; Gender - Female; Emotion - Angry 

path = "../input/speechdataset/my-Audio-Dataset/Emotions/Angry/Angry.02.male.01.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)



# audio wave

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.waveplot(X, sr=sample_rate)

plt.title('Audio sampled at 44100 hrz')



# MFCC

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.specshow(mfcc, x_axis='time')

plt.ylabel('MFCC')

plt.colorbar()



ipd.Audio(path)
# Importing required libraries 

# Keras

import keras

from keras import regularizers

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model, model_from_json

from keras.layers import Dense, Embedding, LSTM

from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.utils import np_utils, to_categorical

from keras.callbacks import ModelCheckpoint



# sklearn

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# Other  

import librosa

import librosa.display

import json

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import seaborn as sns

import glob 

import os

import pickle

import IPython.display as ipd  # To play sound in the notebook
# lets pick up the meta-data that we got from our first part of the Kernel

ref = pd.read_csv("../input/pathsdatanew/Angry.csv")

ref.tail()
# Note this takes a couple of minutes (~10 mins) as we're iterating over 4 datasets 

df = pd.DataFrame(columns=['feature'])



# loop feature extraction over the entire dataset

counter=0

for index,path in enumerate(ref.path):

    X, sample_rate = librosa.load(path

                                  , res_type='kaiser_fast'

                                  ,duration=30

                                  ,sr=44100

                                  ,offset=0.5

                                 )

    sample_rate = np.array(sample_rate)

    

    # mean as the feature. Could do min and max etc as well. 

    mfccs = np.mean(librosa.feature.mfcc(y=X, 

                                        sr=sample_rate, 

                                        n_mfcc=13),

                    axis=0)

    df.loc[counter] = [mfccs]

    counter=counter+1   



# Check a few records to make sure its processed successfully

print(len(df))

df.head()
# Now extract the mean bands to its own feature columns

df = pd.concat([ref,pd.DataFrame(df['feature'].values.tolist())],axis=1)

df[:5]
# replace NA with 0

df=df.fillna(0)

print(df.shape)

df[:5]
# Split between train and test 

X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','label','source'],axis=1)

                                                    , df.label

                                                    , test_size=0.25

                                                    , shuffle=True

                                                    , random_state=42

                                                   )



# Lets see how the data present itself before normalisation 

X_train[10:6]
# Lts do data normalization 

mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0)



X_train = (X_train - mean)/std

X_test = (X_test - mean)/std



# Check the dataset now 

X_train[6:10]
# Lets few preparation steps to get it into the correct format for Keras 

X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)



# one hot encode the target 

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))

y_test = np_utils.to_categorical(lb.fit_transform(y_test))



print(X_train.shape)

print(lb.classes_)

#print(y_train[0:10])

#print(y_test[0:10])



# Pickel the lb object for future use 

filename = 'label'

outfile = open(filename,'wb')

pickle.dump(lb,outfile)

outfile.close()
X_train = np.expand_dims(X_train, axis=2)

X_test = np.expand_dims(X_test, axis=2)

X_train.shape
# New model

model = Sequential()

model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns

model.add(Activation('relu'))

model.add(Conv1D(256, 8, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(14)) # Target class number

model.add(Activation('softmax'))

opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

opt = keras.optimizers.Adam(lr=0.0001)

#opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

model_history=model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_test, y_test))