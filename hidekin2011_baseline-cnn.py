# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





## Python

import os

import random

import sys





## Package

import glob 

import keras

import IPython.display as ipd

import librosa

import librosa.display

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import plotly.graph_objs as go

import plotly.offline as py

import plotly.tools as tls

import seaborn as sns

import scipy.io.wavfile

import tensorflow as tf

py.init_notebook_mode(connected=True)





## Keras

from keras import regularizers

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger

from keras.models import Model, Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.utils import np_utils

from keras.utils import to_categorical



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



## Sklearn

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder





## Rest

from scipy.fftpack import fft

from scipy import signal

from scipy.io import wavfile

from tqdm import tqdm



input_duration=3

# % pylab inline



import os



import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import keras

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Embedding

from keras.layers import LSTM

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.layers import Input, Flatten, Dropout, Activation

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.models import Model

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix



from keras import regularizers



import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import IPython.display as ipd
# Diretório das bases

root_dir = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/" 



fname = root_dir + 'Actor_03/03-01-06-02-01-02-03.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Exemplo de áudio

ipd.Audio(fname)
y, sr = librosa.load(fname)

sr
len(y)
def extract_mfcc(wav_file_name):

    '''This function extracts mfcc features and obtain the mean of each dimension

    Input : path_to_wav_file

    Output: mfcc_features'''

    y, sr = librosa.load(wav_file_name)



    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)



    return mfccs
##### load radvess speech data #####

root_dir = "../input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

# root_dir = "../input/audio_speech_actors_01-24/"

# actor_dir = os.listdir("../input/audio_speech_actors_01-24/")

radvess_speech_labels = []

ravdess_speech_data = []

for actor_dir in sorted(os.listdir(root_dir)):

    actor_name = os.path.join(root_dir, actor_dir)

    for file in os.listdir(actor_name):

        radvess_speech_labels.append(int(file[7:8]) - 1)

        wav_file_name = os.path.join(root_dir, actor_dir, file)

        ravdess_speech_data.append(extract_mfcc(wav_file_name))
#### convert data to array and make labels categorical

ravdess_speech_data_array = np.asarray(ravdess_speech_data)

labels = np.array(radvess_speech_labels)

ravdess_speech_data_array.shape
len(ravdess_speech_data_array[4])
df = pd.DataFrame()

df['feature'] = ravdess_speech_data



df.head()
df3 = pd.DataFrame(df['feature'].values.tolist())

df3 = df3.fillna(0)

df3.head()
sr
df3.shape
def map_emotion(y):

    

    '''

        Função para transformar label segmentados em: neutro,positivo e negativo, sendo considerado

        

        

    '''

    if(y == 0):

        return 'neutro'

    elif(y == 1):

        return 'positivo'

    elif(y == 2):

        return 'positivo'

    elif(y == 3):

        return 'negativo'

    elif(y == 4):

        return 'negativo'

    elif(y == 5):

        return 'negativo'

    elif(y == 6):

        return 'negativo'

    else:

        return 'neutro'
labels_low = np.array([map_emotion(v) for v in labels])

labels_low.shape
labels_low
df3['target'] = labels_low


from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df3.drop('target', axis = 1), df3['target'], test_size = 0.2, stratify = df3['target'], random_state = 42)





from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

y_train = lb.fit_transform(y_train)

y_test = lb.fit_transform(y_test)
from sklearn.metrics import accuracy_score

lr = LogisticRegression(random_state = 42)

lr.fit(X_train, y_train)



accuracy_score(y_test ,lr.predict(X_test))
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)



accuracy_score(y_test ,lr.predict(X_test))
rf = RandomForestClassifier(n_estimators = 50, max_depth = 20 , random_state = 42)

rf.fit(X_train, y_train)



accuracy_score(y_test ,rf.predict(X_test))
rf.predict(X_test)
import pickle



with open('lb.pickle', 'wb') as handle:

    pickle.dump(lb, handle,)

    

with open('rf.pickle', 'wb') as handle:

    pickle.dump(rf, handle,)


from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder



X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)



lb = LabelEncoder()



y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)
y_train[:5]
X_train.shape
print('Pad sequences')

x_traincnn =np.expand_dims(X_train, axis=2)

x_testcnn= np.expand_dims(X_test, axis=2)
# Set up Keras util functions



from keras import backend as K



def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision





def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall





def fscore(y_true, y_pred):

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0



    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    f_score = 2 * (p * r) / (p + r + K.epsilon())

    return f_score



def get_lr_metric(optimizer):

    def lr(y_true, y_pred):

        return optimizer.lr

    return lr
# New model

model = Sequential()

model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))

model.add(Activation('relu'))

model.add(Conv1D(256, 8, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size=(3)))

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

model.add(MaxPooling1D(pool_size=(3)))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Flatten())

# Edit according to target class no.

model.add(Dense(3))

model.add(Activation('softmax'))

opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
# Compile your model



model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# Model Training



lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)

# Please change the model name accordingly.

mcp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')

cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700,

                     validation_data=(x_testcnn, y_test), callbacks=[mcp_save, lr_reduce])