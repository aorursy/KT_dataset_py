from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from scipy.fftpack import fft,fftfreq
import scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import wave

#adicionados
from scipy.io import wavfile as wav
import struct


#add2
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.utils import Sequence
%matplotlib inline  
import gc
import pickle
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from numpy import random
import librosa
import numpy as np
import glob
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder

# Any results you write to the current directory are saved as output.
def extract_features(signal):
    return [       
            librosa.feature.chroma_stft(signal).mean(),
            librosa.feature.chroma_cqt(signal).mean(),
            librosa.feature.chroma_cens(signal).mean(),
            librosa.feature.mfcc(signal).mean(),
            librosa.feature.rms(signal).mean(),
            librosa.feature.spectral_centroid(signal).mean(),
            librosa.feature.spectral_bandwidth(signal).mean(),
            librosa.feature.spectral_contrast(signal).mean(),
            librosa.feature.spectral_flatness(signal).mean(),
            librosa.feature.zero_crossing_rate(signal).mean()
    ]
data = pd.read_csv("../input/urbansound8k/UrbanSound8K.csv")
data.head(10)
data_copy = data.copy()
data_copy.head(10)
fulldatasetpath = '../input/urbansound8k/'

full_path = []


for index, row in tqdm(data.iterrows()):
    filename = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    full_path.append(filename)

data["full_path"] = full_path
data.head(10)
data = data.drop(columns=["slice_file_name", "class"])
data.head(10)
all_audio_signals = [librosa.load(p)[0] for p in data["full_path"]]

#extract features
all_audio_signals_features = np.array([extract_features(x) for x in all_audio_signals])

#normalizar features
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
transformed_features = scaler.fit_transform(all_audio_signals_features)
transformed_features[np.isnan(transformed_features)]=0
dfFeatures = pd.DataFrame(transformed_features)
dfFeatures

data["feature_0"] = dfFeatures[0]
data["feature_1"] = dfFeatures[1]
data["feature_2"] = dfFeatures[2]
data["feature_3"] = dfFeatures[3]
data["feature_4"] = dfFeatures[4]
data["feature_5"] = dfFeatures[5]
data["feature_6"] = dfFeatures[6]
data["feature_7"] = dfFeatures[7]
data["feature_8"] = dfFeatures[8]
data["feature_9"] = dfFeatures[9]

data = data.drop(columns=["full_path", "fold"])
def get_model(num_features):
    
    model = Sequential()
    
    model.add(Dense(100, activation='relu', input_dim=(num_features)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = get_model(10)
model.summary()
target = data['classID'].copy()
data = data.drop(columns = ['classID', 'fsID', 'start', 'end', 'salience'])
data.head(10)

# split train valid test
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)
early_stopping_monitor = EarlyStopping(patience=3)
model.fit(x=X_train, y=y_train, steps_per_epoch = len(X_train), 
          validation_data=(X_val,y_val), validation_steps = len(X_val), epochs=100, verbose=1, callbacks=[early_stopping_monitor])
def get_result(probabilities) : 
    
    mapping = {
        0 : 'air_conditioner',
        1 : 'car_horn',
        2 : 'children_playing',
        3 : 'dog_bark',
        4 : 'drilling',
        5 : 'engine_idling',
        6 : 'gun_shot',
        7 : 'jackhammer',
        8 : 'siren',
        9 : 'street_music'
    }
    
    result_probability = max(probabilities)
    index_position = np.where(probabilities == result_probability)
    
    actual = mapping.get(index_position[0][0])
    print("Classificado como: " + actual)
    return actual
def get_result_index(probabilities) : 
    
    result_probability = max(probabilities)
    index_position = np.where(probabilities == result_probability)
    return index_position[0][0]
result = model.predict(X_test)
from sklearn.metrics import accuracy_score
target_x_test = [get_result_index(x) for x in result]
accuracy = accuracy_score(y_test, target_x_test)
print("Acurácia: ", accuracy)
df = pd.DataFrame(columns = ['Classificado como', 'Classificação real']) 
df.head(10)
mapping = {
        0 : 'air_conditioner',
        1 : 'car_horn',
        2 : 'children_playing',
        3 : 'dog_bark',
        4 : 'drilling',
        5 : 'engine_idling',
        6 : 'gun_shot',
        7 : 'jackhammer',
        8 : 'siren',
        9 : 'street_music'
    }
dataframe_result = pd.DataFrame(result)
dataframe_result.head(10)
classificado_como = []

for index, row in dataframe_result.iterrows():
    classificado_como.append(get_result(row))

df['Classificado como'] = classificado_como
df['Classificação real'] = [mapping.get(index) for index in y_test]
pd.set_option('display.max_rows', df.shape[0]+1)
print(df)

def predict_audio_with_multiple_class(path, train_model):
    
    audio = librosa.load(path)[0]    
    splited_audio = librosa.effects.split(audio)
    audio_features = [extract_features(audio[x[0]:x[1]]) for x in splited_audio]
    transformed_audio_features = scaler.fit_transform(audio_features)
    results = train_model.predict(transformed_audio_features)
    dataframe_result = pd.DataFrame(results)
    classificado_como = []

    for index, row in dataframe_result.iterrows():
        classificado_como.append(get_result(row))
        
    return classificado_como
fs_exemplo, data_exemplo = wav.read('../input/exemplo/exemplo.wav')
ipd.Audio(data_exemplo, rate = fs_exemplo)
predict_audio_with_multiple_class('../input/exemplo/exemplo.wav', model)
fs_exemplo_dois, data_exemplo_dois = wav.read('../input/exemplo/exemplo2.wav')
ipd.Audio(data_exemplo_dois, rate = fs_exemplo_dois)
predict_audio_with_multiple_class('../input/exemplo/exemplo2.wav', model)
fs_exemplo_tres, data_exemplo_tres = wav.read('../input/exemplo/exemplo3.wav')
ipd.Audio(data_exemplo_tres, rate = fs_exemplo_tres)
predict_audio_with_multiple_class('../input/exemplo/exemplo3.wav', model)

