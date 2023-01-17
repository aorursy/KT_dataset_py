# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
from memory_profiler import memory_usage
import os
from glob import glob

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from scipy.fftpack import fft,fftfreq
import numpy, scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import keras
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Divide a amostra em bases de treino, validação e teste
dataset = pd.read_csv('../input/urbansound8k/UrbanSound8K.csv')

dataset_size = len(dataset)
train_size = int(dataset_size*0.7)
valid_size = int(dataset_size*0.15)
test_size = int(dataset_size*0.15)

print('train size: ' + str(train_size))
print('valid size: ' + str(valid_size))
print('test size: ' + str(test_size))

train_data = dataset[0:train_size].copy().reset_index(drop=True)
valid_data = dataset[train_size:(train_size+valid_size)].copy().reset_index(drop=True)
test_data = dataset[(train_size+valid_size):(train_size+valid_size+test_size)].copy().reset_index(drop=True)

print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)
#pegar apenas nome do arquivo
train_data['slice_file_name'][0].split('.')[0]
#função utilizada para extrair as features
def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0, 0],
        librosa.feature.spectral_centroid(signal)[0, 0],
    ]
#extrai features da amostra de treino
train_features = []
fulldatasetpath = '../input/urbansound8k/'
for _, row in tqdm(train_data.iterrows()):
    filename = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    train_signals = librosa.load(filename) #faz a leitura do áudio

    class_label = row["class"] #salva qual a classe do audio

    data = extract_features(train_signals[0]) #extrai as features
    
    train_features.append([data, class_label]) # salva dentro do test_features as features e a classe do audio, duas informacoes que sao necessarias para o treinamento
#extrai features da amostra de validação
valid_features = []
fulldatasetpath = '../input/urbansound8k/'
for _, row in tqdm(valid_data.iterrows()):
    filename = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    valid_signals = librosa.load(filename) #faz a leitura do áudio

    class_label = row["class"] #salva qual a classe do audio

    data = extract_features(valid_signals[0]) #extrai as features
    
    valid_features.append([data, class_label]) # salva dentro do test_features as features e a classe do audio, duas informacoes que sao necessarias para o treinamento
#extrai features da amostra de test
test_features = []
fulldatasetpath = '../input/urbansound8k/'
for _, row in tqdm(test_data.iterrows()):
    filename = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    test_signals = librosa.load(filename) #faz a leitura do áudio

    class_label = row["class"] #salva qual a classe do audio

    data = extract_features(test_signals[0]) #extrai as features
    
    test_features.append([data, class_label]) # salva dentro do test_features as features e a classe do audio, duas informacoes que sao necessarias para o treinamento
# Utilizado para normalização
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

# Normaliza os dados de treino
feat_transf = []
feat_transf = scaler.fit_transform([f[0] for f in train_features])

train_features_norm = []
for i in range(len(train_features)):
    train_features_norm.append([feat_transf[i], train_features[i][1]])
train_features_norm



# Normaliza os dados de validação
feat_transf = []
feat_transf = scaler.fit_transform([f[0] for f in valid_features])

valid_features_norm = []
for i in range(len(valid_features)):
    valid_features_norm.append([feat_transf[i], valid_features[i][1]])
valid_features_norm



# Normaliza os dados de teste
feat_transf = []
feat_transf = scaler.fit_transform([f[0] for f in test_features])


test_features_norm = []
for i in range(len(test_features)):
    test_features_norm.append([feat_transf[i], test_features[i][1]])
test_features_norm



# Transformo os dados de treino em dataframe
df_train = pd.DataFrame({'v1':[],'v2':[],'classe':[]})
for i in range(len(train_features_norm)):
    df_train = df_train.append({'v1': train_features_norm[i][0][0], 'v2': train_features_norm[i][0][1], 'classe': train_features_norm[i][1]}, ignore_index=True)
df_train
#transformo os dados de validação em data frame
df_valid = pd.DataFrame({'v1':[],'v2':[],'classe':[]})
for i in range(len(valid_features_norm)):
    df_valid = df_valid.append({'v1': valid_features_norm[i][0][0], 'v2': valid_features_norm[i][0][1], 'classe': valid_features_norm[i][1]}, ignore_index=True)
df_valid
# transformo os dados de teste em dataframe
df_test = pd.DataFrame({'v1':[],'v2':[],'classe':[]})
for i in range(len(test_features_norm)):
    df_test = df_test.append({'v1': test_features_norm[i][0][0], 'v2': test_features_norm[i][0][1], 'classe': test_features_norm[i][1]}, ignore_index=True)
df_test
df_test.classe
# utilizo o label enconder em cada uma das três amostras

le = LabelEncoder()
train_classes_encoder_int = le.fit_transform(df_train.classe)
valid_classes_encoder_int = le.fit_transform(df_valid.classe)
test_classes_encoder_int = le.fit_transform(df_test.classe)



train_classes_encoder = keras.utils.to_categorical(train_classes_encoder_int)
valid_classes_encoder = keras.utils.to_categorical(valid_classes_encoder_int)
test_classes_encoder = keras.utils.to_categorical(test_classes_encoder_int)
#print(train_classes_encoder[190])
#(df_test.classe[190])
# modelo utilizado com as 10 categorias de resposta com função de ativação softmax e loo categorical crossentropy
def get_dense_model(num_features):
    dnn_model = Sequential()
    dnn_model.add(Dense(256, activation='relu', input_shape=(num_features,)))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(256, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(256, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(10))
    dnn_model.add(Activation('softmax'))

    dnn_model.compile(loss='categorical_crossentropy', optimizer='adam',                 
                      metrics=['accuracy'])
    return dnn_model
model = get_dense_model(2)
model.summary()
data_train = df_train[['v1','v2']]
data_valid = df_valid[['v1','v2']]
data_test = df_test[['v1','v2']]

# train_classes_encoder
len(data_train)
#y_treino = keras.utils.to_categorical(train_classes_encoder, num_classes=10, dtype='float32')
train_classes_encoder
#y_treino = tf.keras.utils.to_categorical(y, num_classes=None, dtype='float32')
batch_size = 32
STEP_SIZE_TRAIN=len(data_train)//batch_size
STEP_SIZE_VALID=len(data_valid)//batch_size
model.fit(x=data_train, y=train_classes_encoder, steps_per_epoch = STEP_SIZE_TRAIN, 
          validation_data=(data_valid,valid_classes_encoder), validation_steps = STEP_SIZE_VALID, epochs=20, verbose=1)
STEP_SIZE_TEST=len(data_test)//batch_size
model.evaluate(x=data_test, y=test_classes_encoder, steps=STEP_SIZE_TEST, verbose=1)
predito_teste = model.predict(x=data_test)
predito_teste
import os
for dirname, _, filenames in os.walk('/kaggle/input/audio-1'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Pega os arquivos de exemplo
exe1_train_signals = librosa.load('../input/audio-1/exemplo.wav')
exe2_train_signals = librosa.load('../input/audio-2/exemplo2.wav')
exe3_train_signals = librosa.load('../input/audio-3/exemplo3.wav')

# Extrai as fitures deles
exe1_data = extract_features(exe1_train_signals[0])
exe2_data = extract_features(exe2_train_signals[0])
exe3_data = extract_features(exe3_train_signals[0])

# estranho
exe2_data
df_exe1_data = pd.DataFrame({'v1':[exe1_data[0]],'v2':[exe1_data[1]]})
df_exe2_data = pd.DataFrame({'v1':[exe2_data[0]],'v2':[exe2_data[1]]})
df_exe3_data = pd.DataFrame({'v1':[exe3_data[0]],'v2':[exe3_data[1]]})
# Previsão exemplo 1
model.predict(df_exe1_data)# Equivale a dog_bark
# Previsão exemplo 2
model.predict(df_exe2_data)
# Previsão exemplo 3
model.predict(df_exe3_data)# Equivale a dog_bark