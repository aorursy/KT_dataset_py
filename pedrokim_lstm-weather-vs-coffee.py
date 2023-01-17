# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot
from datetime import timedelta
from datetime import datetime

from keras.utils import to_categorical

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
full_price = pd.read_csv('../input/coffee-price-noweekend/price_no_weekend.csv')
filled_price = pd.read_csv('../input/coffeevsweather/price_status.csv')
station31 = pd.read_csv('../input/weather/training_data_beta_31.csv')
station32 = pd.read_csv('../input/weather/training_data_beta_32.csv')
station36 = pd.read_csv('../input/weather/training_data_beta_36.csv')


sc = MinMaxScaler()
oh = OneHotEncoder()
def create_lstm_datetime(X, Y, look_back):
    dataX, dataY = [], []
    ncol = len(X.columns)
    
    for i, row in Y.iterrows():
        #a.drop(columns='Data')
        date = row['Data']
        b = row.iloc[1:]
        a = X.loc[ (X['Data'] > date-timedelta(days = look_back)) &  (X['Data'] < date) ].copy()                                                         
        #dataX.append(a.drop(columns='Data').values)
        
        if a.shape == (look_back-1,ncol):
            dataX.append(a.drop(columns='Data').values)
            dataY.append(b)
            

    return np.array(dataX), np.expand_dims(np.array(dataY),-1)

def create_dataset(X, Y, look_back):
    dataX, dataY = [], []
    for i in range(len(Y)-look_back-1):
        a = X[i:(i+look_back)]
        dataX.append(a)

        #b = Y[i+1:(i+look_back+1)]
        b = Y[(i+look_back)][0]
        dataY.append(b)
    #print(dataX)
    return np.array(dataX), np.array(dataY)

def status_def(full_price):
    v0 = 0
    status_lst = []
    previous = 'Alta'
    for i, row in full_price.iterrows():
        sample = row['Valor'] - v0
        if sample>0:
            result = 1
        elif sample == 0:
            result == previous
        else:
            result = 0

        v0 = row['Valor']
        previous = result
        status_lst.append(result)    
    return status_lst
### Output Classes


full_price['Status'] = status_def(full_price)
### Input Prep

full_price['Data'] = pd.to_datetime(full_price['Data'], format='%d/%m/%Y')
heading = ['Precipitacao', 'TempMaxima', 'TempMinima', 'Insolacao', 'Evaporacao Piche', 'Temp Comp Media', 'Umidade Relativa Media', 'Velocidade do Vento Media']

#start_date = '31/12/2002'
start_date = '01/01/2003'
end_date = '01/01/2013'
price = full_price.loc[ (full_price['Data'] > datetime.strptime(start_date, "%d/%m/%Y")) &  (full_price['Data'] < datetime.strptime(end_date, "%d/%m/%Y")) ]                                                          

# Weather
df_full = pd.DataFrame([], columns=heading)
for col in heading:
    x = station31[col].values.astype(float)
    df_full[col] = sc.fit_transform(np.expand_dims(x, -1)).T[0]
    x = station32[col].values.astype(float)
    df_full[col+'2'] = sc.fit_transform(np.expand_dims(x, -1)).T[0]
    x = station36[col].values.astype(float)
    df_full[col+'3'] = sc.fit_transform(np.expand_dims(x, -1)).T[0]

df_full['Data'] = station32['Data']
X_full_W = df_full


### Output Preparations

#Y = oh.fit_transform(np.expand_dims(price['Status'], -1)).toarray()### Boolean up or down
#print(Y[:, 0])

price.head

price['Data'] = pd.to_datetime(price['Data'], format='%d/%m/%Y')
X_full_W['Data'] = pd.to_datetime(X_full_W['Data'], format='%d/%m/%Y')


X_lstm, Y_lstm = create_lstm_datetime(X_full_W, price, 14)
np.expand_dims(X_lstm,-1).shape

X_lstm, Y_lstm = create_lstm_datetime(X_full_W, price, 21)
#np.expand_dims(X_lstm,-1).shape
Y_lstm_class = Y_lstm[:, 1].astype('int32') 
Y_lstm_val = sc.fit_transform(Y_lstm[:, 0])
Y_lstm_val = Y_lstm[:, 0].astype('float64') 
input_shape = X_lstm[1].shape
inputA = Input(shape = input_shape)

#x = LSTM(2, input_shape = input_shape)(inputA) ### Camada de LSTM, 2 memórias
x = LSTM(14, input_shape = input_shape, dropout = 0.3)(inputA) ### Camada de LSTM, 14 memórias
#x = LSTM(21, input_shape = input_shape)(inputA) ### Camada de LSTM, 21 memórias
#x = LSTM(31, input_shape = input_shape)(inputA) ### Camada de LSTM, 31 memórias
#x = LSTM(41, input_shape = input_shape, dropout = 0.1)(inputA) ### Camada de LSTM, 41 memórias
#x = LSTM(41, input_shape = input_shape, dropout = 0.2)(inputA) ### Camada de LSTM, 41 memórias

############## Output setting
### Boolean Up vs Down
x = Dense(5, activation = 'sigmoid')(x)
z = Dense(1, activation = 'sigmoid')(x)        ### Camada densa de classificação
model = Model(inputs=[inputA], outputs=z)
optimizer = K.optimizers.Adam(lr=0.005)
model.compile( loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])

model.summary()





batch_size = 64
max_epochs = 1000

#h = model.fit(x = x_train, y = Y_train, batch_size= batch_size, epochs= max_epochs, verbose=1, validation_split=0.1)
h = model.fit(x = X_lstm, y = Y_lstm_class, batch_size= batch_size, epochs= max_epochs, verbose=1, validation_split=0.3)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
input_shape = X_lstm[1].shape
inputA = Input(shape = input_shape)

#x = LSTM(2, input_shape = input_shape)(inputA) ### Camada de LSTM, 2 memórias
x = LSTM(7, input_shape = input_shape, dropout = 0.3)(inputA) ### Camada de LSTM, 14 memórias

############## Output setting


### Numeric Value 
z = Dense(1)(x)
model = Model(inputs=[inputA], outputs=z)
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
model.summary()

batch_size = 64
max_epochs = 1000

h = model.fit(x = X_lstm, y = Y_lstm_val, batch_size= batch_size, epochs= max_epochs, verbose=1, validation_split=0.3)
plt.plot(h.history['mse'])
plt.plot(h.history['val_mse'])
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
division = int(len(X_lstm) * 0.7)

x_train = X_lstm[0:division]
x_test = X_lstm[division:]

Y_train =  Y_lstm_val[0:division]
Y_test = Y_lstm_val[division:]

np.expand_dims(x_test,-1).shape
predictions = model.predict(x_test)
print( x_test.shape)
plt.plot(predictions)
plt.plot(Y_test)
plt.show()
x_test = X_lstm[division+110 : division+200]
Y_test = Y_lstm_val[division+110 : division+200]
predictions = model.predict(x_test)

plt.plot(predictions)
plt.plot(Y_test)
plt.show()
start_date = '01/01/2002'
end_date = '01/01/2013'
filled_price['Data'] = pd.to_datetime(filled_price['Data'], format='%d/%m/%Y')
price_filled = filled_price.loc[ (filled_price['Data'] > datetime.strptime(start_date, "%d/%m/%Y")) &  (filled_price['Data'] < datetime.strptime(end_date, "%d/%m/%Y")) ]                                                          


Y = oh.fit_transform(np.expand_dims(price_filled['Status'], -1)).toarray()### Boolean up or down

df_day_filled = X_full_W
df_day_filled['Valor'] = price_filled['Valor']
X = df_day_filled.drop(columns='Data').to_numpy()
X_lstm_fill_class, Y_lstm_fill_class = create_dataset(X, np.expand_dims(Y[:, 0], -1), 21)

Y_lstm_fill_class.shape
input_shape = X_lstm_fill_class[1].shape
inputA = Input(shape = input_shape)

#x = LSTM(7, input_shape = input_shape)(inputA) ### Camada de LSTM, 7 memórias
x = LSTM(14, input_shape = input_shape, dropout = 0.5)(inputA) ### Camada de LSTM, 7 memórias
#x = LSTM(21, input_shape = input_shape)(inputA) ### Camada de LSTM, 21 memórias
#x = LSTM(31, input_shape = input_shape)(inputA) ### Camada de LSTM, 31 memórias
#x = LSTM(41, input_shape = input_shape)(inputA) ### Camada de LSTM, 41 memórias
#x = LSTM(41, input_shape = input_shape, dropout = 0.2)(inputA) ### Camada de LSTM, 41 memórias

############## Output setting
### Boolean Up vs Down
x = Dense(5, activation = 'sigmoid')(x)
z = Dense(1, activation = 'sigmoid')(x)        ### Camada densa de classificação
model = Model(inputs=[inputA], outputs=z)
optimizer = K.optimizers.Adam(lr=0.005)
model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()
batch_size = 64
max_epochs = 1000

h = model.fit(x = X_lstm_fill_class, y = Y_lstm_fill_class, batch_size= batch_size, epochs= max_epochs, verbose=1, validation_split=0.3)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#x = , y = 

division = int(len(X_lstm_fill_class) * 0.7)


#division = len(X_lstm_fill_class) - 1095

x_train = X_lstm_fill_class[0:division]
x_test = X_lstm_fill_class[division:]

Y_train =  Y_lstm_fill_class[0:division]
Y_test = Y_lstm_fill_class[division:]

#np.expand_dims(x_test[0],-1)
#predictions = model.predict(x_test[0:100])
#classes = np.argmax(predictions, axis = 1)

eval_test1 = model.evaluate(x_train, Y_train, verbose=0)
print("Erro médio do treino: Perda {0:.4f}, Acc. {1:.4f}".format(eval_test1[0], eval_test1[1]))

predictions = model.predict(x_test)
print( x_test.shape)
plt.plot(predictions, 'r+')

plt.show()
x_train
price_week = price_filled.set_index('Data').resample('7D').mean().reset_index()
price_week['Status'] = status_def(price_week)
df_week = X_full_W.set_index('Data').resample('7D').mean().reset_index()
X_week = df_week.drop(columns='Data').to_numpy()
df_week['Valor'] = price_week['Valor']
X_week_valor = df_week.drop(columns='Data').to_numpy()

Y = oh.fit_transform(np.expand_dims(price_week['Status'], -1)).toarray()### Boolean up or down

#X = X_full_W.drop(columns='Data').to_numpy()
X_lstm_fill_class_week, Y_lstm_fill_class_week = create_dataset(X_week_valor, np.expand_dims(Y[:, 0], -1), 8)
input_shape = X_lstm_fill_class_week[1].shape
inputA = Input(shape = input_shape)

x = LSTM(7, input_shape = input_shape)(inputA) ### Camada de LSTM, 7 memórias
#x = LSTM(14, input_shape = input_shape, dropout = 0.5)(inputA) ### Camada de LSTM, 7 memórias
#x = LSTM(21, input_shape = input_shape)(inputA) ### Camada de LSTM, 21 memórias
#x = LSTM(31, input_shape = input_shape)(inputA) ### Camada de LSTM, 31 memórias
#x = LSTM(41, input_shape = input_shape)(inputA) ### Camada de LSTM, 41 memórias
#x = LSTM(41, input_shape = input_shape, dropout = 0.2)(inputA) ### Camada de LSTM, 41 memórias

############## Output setting
### Boolean Up vs Down
x = Dense(5, activation = 'sigmoid')(x)
z = Dense(1, activation = 'sigmoid')(x)        ### Camada densa de classificação
model = Model(inputs=[inputA], outputs=z)
optimizer = K.optimizers.Adam(lr=0.005)
model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()
batch_size = 32
max_epochs = 500


h = model.fit(x = X_lstm_fill_class_week, y = Y_lstm_fill_class_week, batch_size= batch_size, epochs= max_epochs, verbose=1, validation_split=0.3)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
division = int(len(X_lstm_fill_class) * 0.7)


x_train = X_lstm_fill_class[0:division]
x_test = X_lstm_fill_class[division:]

Y_train =  Y_lstm_fill_class[0:division]
Y_test = Y_lstm_fill_class[division:]
Y =  np.expand_dims(price_week["Valor"].to_numpy(), -1)

X_lstm_fill_reg_week, Y_lstm_fill_reg_week = create_dataset(X_week_valor,Y, 8)
Y_lstm_fill_reg_week = np.expand_dims(Y_lstm_fill_reg_week, -1)
type(Y_lstm_fill_reg_week[0][0])
type(X_lstm_fill_reg_week[0][0][0])

input_shape = X_lstm_fill_reg_week[1].shape
inputA = Input(shape = input_shape)

#x = LSTM(14, input_shape = input_shape, dropout = 0.3)(inputA) ### Camada de LSTM, 14 memórias
x = LSTM(14, input_shape = input_shape)(inputA) ### Camada de LSTM, 14 memórias
#x = Flatten()(inputA)
############## Output setting


### Numeric Value 
#x = Dense(256, activation = 'tanh')(x)
#x = Dense(7, activation = 'relu')(x)
z = Dense(1)(x)
model = Model(inputs=[inputA], outputs=z)
model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae'])
model.summary()
batch_size = 64
max_epochs = 500

h = model.fit(x = X_lstm_fill_reg_week, y = Y_lstm_fill_reg_week, batch_size= batch_size, epochs= max_epochs, verbose=1, validation_split=0.3)
#h = model.fit(x = X_lstm_fill_reg_week, y = Y_lstm_fill_reg_week, batch_size= batch_size, epochs= max_epochs, verbose=1)
plt.plot(h.history['mse'])
plt.plot(h.history['val_mse'])
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
division = int(len(X_lstm_fill_reg_week) * 0.7)

x_train = X_lstm_fill_reg_week[0:division]
x_test = X_lstm_fill_reg_week[division:]

Y_train =  Y_lstm_fill_reg_week[0:division]
Y_test = Y_lstm_fill_reg_week[division:]

predictions = model.predict(x_test)
#predictions = model.predict(X_lstm_fill_reg_week)

plt.plot(predictions, "r")
plt.plot(Y_test)
plt.grid(True)
plt.legend(['Previsão', 'Real'], loc='upper left')
plt.show()
bot, top = 140, 152
plt.plot(predictions[bot:top], "r-o")
plt.plot(Y_test[bot:top], "-o")
plt.grid(True)
plt.legend(['Previsão', 'Real'], loc='upper left')
plt.show()
Y =  np.expand_dims(price_week["Valor"].to_numpy(), -1)
X_lstm_week_v, Y_lstm_week_v = create_dataset(Y,Y, 8)
#X_lstm, Y_lstm = create_dataset(X_full_P, Y, 21)
#X_lstm = np.expand_dims(X_lstm, -1)
#X_lstm[1].shape
input_shape = X_lstm[1].shape
inputA = Input(shape = input_shape)


#x = LSTM(7, input_shape = input_shape)(inputA) ### Camada de LSTM, 7 memórias
x = LSTM(14, input_shape = input_shape, dropout = 0.3)(inputA) ### Camada de LSTM, 21 memórias
#x = LSTM(31, input_shape = input_shape)(inputA) ### Camada de LSTM, 31 memórias
#x = LSTM(41, input_shape = input_shape)(inputA) ### Camada de LSTM, 41 memórias
#x = LSTM(41, input_shape = input_shape, dropout = 0.2)(inputA) ### Camada de LSTM, 41 memórias

############## Output setting
### Boolean Up vs Down
z = Dense(1)(x)        ### Camada densa de classificação
model = Model(inputs=[inputA], outputs=z)
optimizer = K.optimizers.Adam(lr=0.005)
model.compile( loss='mse', optimizer= optimizer, metrics=['mse', 'mae'])

model.summary()
batch_size = 64
max_epochs = 500

h = model.fit(x = X_lstm_week_v, y = Y_lstm_week_v, batch_size= batch_size, epochs= max_epochs, verbose=1, validation_split=0.3)
plt.plot(h.history['mse'])
plt.plot(h.history['val_mse'])
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#X_lstm_week_v, Y_lstm_week_v

division = int(len(X_lstm_week_v) * 0.7)

x_train = X_lstm_week_v[0:division]
x_test = X_lstm_week_v[division:]

Y_train =  Y_lstm_week_v[0:division]
Y_test = Y_lstm_week_v[division:]

#predictions = model.predict(x_test)
predictions = model.predict(X_lstm_week_v)
plt.plot(predictions, "--")
plt.plot(Y_lstm_week_v)
plt.show()

eval_test1 = model.evaluate(x_test, Y_test, verbose=0)
#print("Erro médio do teste: Perda {0:.4f}, acuracia {1:.4f}".format(eval_test1[0], eval_test1[1]*100))
print("Erro médio do teste: Acuracia {0:.4f}".format(eval_test1[1]*100))
price.groupby(['Status']).describe()
price.head()
price['Valor'].plot()

