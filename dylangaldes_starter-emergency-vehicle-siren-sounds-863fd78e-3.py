from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd





pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV



from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout

from tensorflow.keras.utils import to_categorical 



from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

# Project Specific Libraries





import os

import librosa

import librosa.display

import glob 

import skimage

dat1, sampling_rate1 = librosa.load('/kaggle/input/firetruck/sound_323.wav')

dat2, sampling_rate2 = librosa.load('/kaggle/input/ambulance/sound_106.wav')
'''1'''

plt.figure(figsize=(20, 10))

D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)

plt.subplot(4, 2, 1)

librosa.display.specshow(D, y_axis='linear')

plt.colorbar(format='%+2.0f dB')

plt.title('Linear-frequency power spectrogram')
''''2'''



path = '/kaggle/input/firetruck/sound_323.wav'

path1 = '/kaggle/input/ambulance/sound_106.wav'

path2 = '/kaggle/input/traffic/sound_406.wav'

data, sampling_rate = librosa.load(path)

plt.figure(figsize=(10, 5))

D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)

plt.subplot(4, 2, 1)

librosa.display.specshow(D, y_axis='linear')

plt.colorbar(format='%+2.0f dB')

plt.title("Firetruck")

    

data, sampling_rate = librosa.load(path1)

plt.figure(figsize=(10, 5))

D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)

plt.subplot(4, 2, 1)

librosa.display.specshow(D, y_axis='linear')

plt.colorbar(format='%+2.0f dB')

plt.title("Ambulance")

    

data, sampling_rate = librosa.load(path2)

plt.figure(figsize=(10, 5))

D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)

plt.subplot(4, 2, 1)

librosa.display.specshow(D, y_axis='linear')

plt.colorbar(format='%+2.0f dB')

plt.title("Traffic")
'''EXAMPLE'''



dat1, sampling_rate1 = librosa.load('/kaggle/input/ambulance/sound_106.wav')

arr = librosa.feature.melspectrogram(y=dat1, sr=sampling_rate1)

arr.shape
# 3

feature = []

label = []

folders = ['ambulance', 'firetruck', 'traffic']

ambulance = []

firetruck = []

traffic = []

for folder in folders:

    if folder == 'ambulance':

        ambulance = os.listdir('/kaggle/input/'+folder)    

    elif folder =='firetruck':

        firetruck = os.listdir('/kaggle/input/'+folder)      

    else:

        traffic = os.listdir('/kaggle/input/'+folder)     



#4

ambulanceCount = len(ambulance)

fireTruckCount = len(firetruck)

trafficCount =  len(traffic)

feature = []

label = []

#Loads all files in folders and extracts their features

def parser():

    for folder in folders:

        if folder == 'ambulance':

            for i in range(ambulanceCount):

                if ambulance[i] != 'sample.py':

                    X, sample_rate = librosa.load('/kaggle/input/'+folder+'/'+ambulance[i], res_type='kaiser_fast')

                    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)

                    feature.append(mels)

                    label.append(1)

        elif folder =='firetruck':

            for i in range(fireTruckCount):

                if firetruck[i] != 'sample.py':

                    X, sample_rate = librosa.load('/kaggle/input/'+folder+'/'+firetruck[i], res_type='kaiser_fast')

                    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)

                    feature.append(mels)

                    label.append(1)

        else:

            for i in range(trafficCount):

                if (traffic[i] != 'sample.py') and (traffic[i] != "sound_601.wav"):

                    X, sample_rate = librosa.load('/kaggle/input/'+folder+'/'+traffic[i], res_type='kaiser_fast')

                    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)

                    feature.append(mels)

                    label.append(0)

            return [feature, label]

    
temp = parser()

temp = np.array(temp)

data = temp.transpose()
X_ = data[:, 0]

Y = data[:, 1]

print(X_.shape, Y.shape)

X = np.empty([600, 128])
for i in range(600):

    X[i] = (X_[i])
Y = to_categorical(Y)
print(X.shape)

print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)
'''5'''

X_train = X_train.reshape(450, 16, 8, 1)

X_test = X_test.reshape(150, 16, 8, 1)
input_dim = (16, 8, 1)
model = Sequential()
print(input_dim)
'''6'''

model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(1024, activation = "tanh"))

model.add(Dense(2, activation = "softmax"))
'''7'''

from keras import backend as K



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', f1_m,precision_m, recall_m])
model.fit(X_train, Y_train, epochs = 90, batch_size = 50, validation_data = (X_test, Y_test))
'''8'''

model.summary()


predictions = model.predict(X_test)

score = model.evaluate(X_test, Y_test)

print(score)
'''9'''

 