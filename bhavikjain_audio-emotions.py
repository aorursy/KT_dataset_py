# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os
import pandas as pd
import glob 
import librosa

import numpy as np
import joblib

X = joblib.load('/kaggle/input/1234567/X.joblib')
y = joblib.load('/kaggle/input/1234567/y.joblib')
X.shape
y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)
x_traincnn.shape, x_testcnn.shape
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv1D(128, 5,padding='same',
                 input_shape=(40,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=100, validation_data=(x_testcnn, y_test))
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
SAVEE = "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"

from tensorflow.keras.models import load_model
new_model = tf.keras.models.load_model('/kaggle/input/model-lstm/model_LSTM (1).h5')
new_model.summary()
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
dir_list = os.listdir(SAVEE)

# parse the filename to get the emotions
emotion=[]
path = []
for i in dir_list:
    if i[-8:-6]=='_a':
        emotion.append('angry')
    elif i[-8:-6]=='_d':
        emotion.append('disgust')
    elif i[-8:-6]=='_f':
        emotion.append('fearful')
    elif i[-8:-6]=='_h':
        emotion.append('happy')
    elif i[-8:-6]=='_n':
        emotion.append('neutral')
    elif i[-8:-6]=='sa':
        emotion.append('sad')
    elif i[-8:-6]=='su':
        emotion.append('surprise')
    else:
        emotion.append('male_error') 
    path.append(SAVEE + i)
    
# Now check out the label count distribution 
SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])
SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)
SAVEE_df
def extract_mfcc(wav_file_name):
    #This function extracts mfcc features and obtain the mean of each dimension
    #Input : path_to_wav_file
    #Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name,duration=3
                                  ,offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    
    return mfccs
data = [] 
for index,path in enumerate(SAVEE_df.path):
    data.append(extract_mfcc(path)) # extract MFCC features/file

print("Finish Loading the Dataset")
data1 = np.asarray(data)
data1.shape
dict = {0 : 'neutral', 1 : 'happy', 2 : 'happy', 3 : 'sad', 4 : 'angry', 5 : 'fearful', 6 : 'disgust', 7 : 'surprised'}
pred = new_model.predict(np.expand_dims(data1,-1))
preds=pred.argmax(axis=1)
new = np.array([dict[x] for x in preds])
new
from sklearn.metrics import accuracy_score
accuracy_score(SAVEE_df.labels, new)
dir_list = os.listdir("/kaggle/input/cremad/AudioWAV/")
dir_list.sort()
print(dir_list[0:10])

CREMA = "/kaggle/input/cremad/AudioWAV/"
gender = []
emotion = []
path = []
female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]

for i in dir_list: 
    part = i.split('_')
    if int(part[0]) in female:
        temp = 'female'
    else:
        temp = 'male'
    gender.append(temp)
    if part[2] == 'SAD' and temp == 'male':
        emotion.append('sad')
    elif part[2] == 'ANG' and temp == 'male':
        emotion.append('angry')
    elif part[2] == 'DIS' and temp == 'male':
        emotion.append('disgust')
    elif part[2] == 'FEA' and temp == 'male':
        emotion.append('fear')
    elif part[2] == 'HAP' and temp == 'male':
        emotion.append('happy')
    elif part[2] == 'NEU' and temp == 'male':
        emotion.append('neutral')
    elif part[2] == 'SAD' and temp == 'female':
        emotion.append('sad')
    elif part[2] == 'ANG' and temp == 'female':
        emotion.append('angry')
    elif part[2] == 'DIS' and temp == 'female':
        emotion.append('disgust')
    elif part[2] == 'FEA' and temp == 'female':
        emotion.append('fearful')
    elif part[2] == 'HAP' and temp == 'female':
        emotion.append('happy')
    elif part[2] == 'NEU' and temp == 'female':
        emotion.append('neutral')
    else:
        emotion.append('Unknown')
    path.append(CREMA + i)
    
CREMA_df = pd.DataFrame(emotion, columns = ['labels'])
CREMA_df['source'] = 'CREMA'
CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)
CREMA_df
x = [] 
for index,path in enumerate(CREMA_df.path):
    x.append(extract_mfcc(path)) # extract MFCC features/file

print("Finish Loading the Dataset")
x = np.asarray(x)
x.shape
y = new_model.predict(np.expand_dims(x,-1))
y=y.argmax(axis=1)
new1 = np.array([dict[x] for x in y])
new1
from sklearn.metrics import accuracy_score
accuracy_score(CREMA_df.labels, new1)
