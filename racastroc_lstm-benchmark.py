import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import numpy as np

import csv

import sys 

import numpy as np

from scipy.stats import randint

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler 

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline 

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn import metrics 

from sklearn.metrics import mean_squared_error,r2_score



import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.utils import to_categorical

from keras.optimizers import SGD 

from keras.callbacks import EarlyStopping

from keras.utils import np_utils

import itertools

from keras import backend as K

from keras.layers import LSTM

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers import Dropout

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/seriestemporales-diplomado/train.txt', sep = ';', index_col=0)

test = pd.read_csv('/kaggle/input/seriestemporales-diplomado/test.txt', sep = ';', index_col=0)
train = train.set_index('Fecha_Hora')
train
test['Fecha_Hora'] = test['Fecha'] + ' ' + test['Hora']
test = test.drop('Fecha', axis = 1)

test = test.drop('Hora', axis = 1)

test.head()
test = test[['Fecha_Hora','Poder_Reactivo_Global', 'Voltaje', 'Intensidad_Global', 'Medida_1',

       'Medida_2', 'Medida_3']]
test = test.set_index('Fecha_Hora')
test
result = [train, test]

dataset = pd.concat(result)
dataset = dataset.replace('?', np.nan)
dataset.head()
import seaborn as sns
sns.heatmap(dataset.isnull())
# Viendo si hay nulos aún

dataset.isnull().sum()
dataset.dtypes
set_test = dataset.iloc[-8760:,:]
set_train = dataset.iloc[:-8760,:]
set_train
set_test
dataset.head()
dataset.shape
droping_list_all=[]

for j in range(0,7):

    if not dataset.iloc[:, j].notnull().all():

        droping_list_all.append(j)   

droping_list_all
set_train = set_train.apply(pd.to_numeric)
set_train.info()
set_train.iloc[:,0]=set_train.iloc[:,0].fillna(set_train.iloc[:,0].mean())
set_train.iloc[:,1]=set_train.iloc[:,1].fillna(set_train.iloc[:,1].mean())
set_train.iloc[:,2]=set_train.iloc[:,2].fillna(set_train.iloc[:,2].mean())
set_train.iloc[:,3]=set_train.iloc[:,3].fillna(set_train.iloc[:,3].mean())
set_train.iloc[:,4]=set_train.iloc[:,4].fillna(set_train.iloc[:,4].mean())
set_train.iloc[:,5]=set_train.iloc[:,5].fillna(set_train.iloc[:,5].mean())
set_train.iloc[:,6]=set_train.iloc[:,6].fillna(set_train.iloc[:,6].mean())
# Viendo si hay nulos aún

set_train.isnull().sum()
set_train.info()
set_train.columns
set_train = set_train[['Intensidad_Global', 'Medida_1', 'Medida_2', 'Medida_3', 'Poder_Reactivo_Global', 'Voltaje',

       'Poder_Activo_Global']]

set_train
set_train['Poder_Activo_Global'].max()
set_test = set_test[['Intensidad_Global', 'Medida_1', 'Medida_2', 'Medida_3', 'Poder_Reactivo_Global', 'Voltaje',

       'Poder_Activo_Global']]

set_test
values = set_train.values 

#Aplicar escalamiento

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)
scaled.shape
values = scaled

n_train_time_start = 1920000

n_train_time = 2000000

train = values[n_train_time_start:n_train_time, :]

test = values[n_train_time:, :]



train_X, train_y = train[:, :-1], train[:, -1]

test_X, test_y = test[:, :-1], test[:, -1]



train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
train_y
def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
#Construye tu propio modelo y entrena LSTM
# Historia de la funcion de perdida

#history = entrenamiento

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
# Hacer una predicción

yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], 6))

# Invertir el escalamiento

inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]

# Invertir el escalamiento para el test_Y

test_y = test_y.reshape((len(test_y), 1))

inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]

# Calcular el RMSE

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
inv_y
inv_yhat
set_test = set_test.apply(pd.to_numeric)
set_test.isnull().sum()
set_test_to_predict = set_test[['Intensidad_Global', 'Medida_1', 'Medida_2', 'Medida_3', 'Poder_Reactivo_Global', 'Voltaje']]
set_test_to_predict
values = set_test_to_predict.values 

scaler_test = MinMaxScaler(feature_range=(0, 1))

scaled_test = scaler_test.fit_transform(values)
to_predict = scaled_test

to_predict = to_predict.reshape((to_predict.shape[0], 1, to_predict.shape[1]))
to_predict.shape
ypredict = model.predict(to_predict)
ypredict
to_predict.shape
to_predict = to_predict.reshape((to_predict.shape[0], 6))

inv_ypredict = np.concatenate((to_predict[:, :], ypredict), axis=1)
inv_ypredict.shape
inv_ypredict = scaler.inverse_transform(inv_ypredict)

ypredict_final = inv_ypredict[:,-1]
inv_ypredict
ypredict_final
prediction = pd.DataFrame(ypredict_final)
plt.plot(prediction)
set_test['Poder_Activo_Global'] = ypredict_final
set_test
enviar = set_test[['Poder_Activo_Global']]
enviar['Fecha_Hora'] = enviar.index
enviar = enviar.reset_index(drop=True)

enviar = enviar[['Fecha_Hora', 'Poder_Activo_Global']]

enviar
enviar.to_csv('Output.csv',index=False)