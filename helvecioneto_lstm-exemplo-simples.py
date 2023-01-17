%matplotlib inline

import warnings

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import datetime

from functools import reduce

import urllib.request

from sklearn.preprocessing import MinMaxScaler



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



warnings.filterwarnings('ignore')
!pip install tensorflow==1.15.0rc3
# Bibliotecas Keras

import tensorflow as tf

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Input, Dense, LSTM, GRU, Embedding, Dropout

from tensorflow.python.keras.optimizers import RMSprop

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
def open_data():

    

    mtal = pd.read_csv('../input/dataset/dados_monte_alegre.csv', delimiter=';')

    belt = pd.read_csv('../input/dataset/dados_belterra.csv', delimiter=';')

    orix = pd.read_csv('../input/dataset/dados_oriximina.csv', delimiter=';')

    

    ### Organização dataset

    mtal[['day','month','year']] = mtal.Data.str.split('/', n = 3, expand = True)

    mtal['day'] = pd.to_numeric(mtal.day)

    mtal['month'] = pd.to_numeric(mtal.month)

    mtal['year'] = pd.to_numeric(mtal.year)

    mtal = mtal.replace([1200,1800],[12,18])



    mtal['datetime'] = mtal[['day','month','year','Hora']].apply(lambda row:

                        datetime.datetime(year=row['year'], month=row['month'],day=row['day'],hour=row['Hora']),axis=1)

    mtal = mtal.drop(columns=['Data','Hora','day','month','year','Unnamed: 10'])

    mtal = mtal.set_index('datetime')

    mtal.sort_values('datetime',ascending=True,inplace=True)

    mtal = mtal['1988':'2018']

    

    belt[['day','month','year']] = belt.Data.str.split('/', n = 3, expand = True)

    belt['day'] = pd.to_numeric(belt.day)

    belt['month'] = pd.to_numeric(belt.month)

    belt['year'] = pd.to_numeric(belt.year)

    belt = belt.replace([1200,1800],[12,18])



    belt['datetime'] = belt[['day','month','year','Hora']].apply(lambda row:

                        datetime.datetime(year=row['year'], month=row['month'],day=row['day'],hour=row['Hora']),axis=1)

    belt = belt.drop(columns=['Data','Hora','day','month','year','Unnamed: 10'])

    belt = belt.set_index('datetime')

    belt.sort_values('datetime',ascending=True,inplace=True)

    belt = belt['1988':'2018']

    

    orix[['day','month','year']] = orix.Data.str.split('/', n = 3, expand = True)

    orix['day'] = pd.to_numeric(orix.day)

    orix['month'] = pd.to_numeric(orix.month)

    orix['year'] = pd.to_numeric(orix.year)

    orix = orix.replace([1200,1800],[12,18])



    orix['datetime'] = orix[['day','month','year','Hora']].apply(lambda row:

                        datetime.datetime(year=row['year'], month=row['month'],day=row['day'],hour=row['Hora']),axis=1)

    orix = orix.drop(columns=['Data','Hora','day','month','year','Unnamed: 10'])

    orix = orix.set_index('datetime')

    orix.sort_values('datetime',ascending=True,inplace=True)

    orix = orix['1988':'2018']

    

    ## Merge entre os datasets

    dfs = [mtal,belt,orix]

    del mtal,belt,orix

    dfs_final = reduce(lambda left,right: pd.merge(left,right, on='datetime'),dfs)



    # Remover colunas

    dfs_final = dfs_final.drop(columns=['Estacao_x','DirecaoVento_x','VelocidadeVento_x','Nebulosidade_x',

                                       'Estacao_y','DirecaoVento_y','VelocidadeVento_y','Nebulosidade_y',

                                       'Estacao','DirecaoVento','VelocidadeVento','Nebulosidade',

                                       'TempBulboUmido_x','TempBulboUmido_y','TempBulboUmido'])



    dfs_final = dfs_final.rename(columns={'TempBulboSeco_x':'TempBS_MTA','UmidadeRelativa_x':'UmidR_MTA','PressaoAtmEstacao_x':'Pres_MTA',

                                          'TempBulboSeco_y':'TempBS_BLT','UmidadeRelativa_y':'UmidR_BLT','PressaoAtmEstacao_y':'Pres_BLT',

                                          'TempBulboSeco':'TempBS_OBD','UmidadeRelativa':'UmidR_OBD','PressaoAtmEstacao':'Pres_OBD'} )

    dfs_final.head()

    

    return dfs_final
def remove_outliers(dfs_final):

    

    ## Remoção de Outliers por quantil

    low = .001

    high = .9999

    quant_df = dfs_final.quantile([low,high])

    

    # Função para correção dos outliers

    filt_df = dfs_final.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 

                                    (x < quant_df.loc[high,x.name])], axis=0)

    

    return filt_df
def imputation(filt_df):

    

    # Media entre Belterra e Oriximina

    filt_df['Pres_MTA'] = filt_df.Pres_MTA.fillna(filt_df[['Pres_BLT','Pres_OBD']].mean(axis=1))

    

    # Media entre Belterra e Oriximina

    filt_df['UmidR_MTA'] = filt_df.UmidR_MTA.fillna(filt_df[['UmidR_BLT','UmidR_OBD']].mean(axis=1))

    

    imp = IterativeImputer(max_iter=10, random_state=0)



    ## Imputação para Temperatura Bulbo Seco

    timp = imp.fit(filt_df.loc[:,['TempBS_MTA','TempBS_BLT','TempBS_OBD']].to_numpy())

    timp = timp.transform(filt_df.loc[:,['TempBS_MTA','TempBS_BLT','TempBS_OBD']].to_numpy())

    

    df_final = filt_df.drop(['TempBS_MTA','TempBS_BLT','TempBS_OBD'], axis=1)

    df_final['TempBS_MTA'] = timp[:,0]

    df_final['TempBS_BLT'] = timp[:,1]

    df_final['TempBS_OBD'] = timp[:,2]

    

     #  Removendo os valores nulos

    df_final = df_final.dropna()



    df_final = df_final.reindex(columns=['TempBS_MTA','UmidR_MTA','Pres_MTA',

               'TempBS_BLT','UmidR_BLT','Pres_BLT',

               'TempBS_OBD','UmidR_OBD','Pres_OBD'])

    

    del filt_df,imp,timp

    

    return df_final
## Normalizacao

def normalization(x_train,x_test,y_train,y_test):

    # objeto para os sinais de entrada

    x_scaler = MinMaxScaler()

    

    x_train_scaled = x_scaler.fit_transform(x_train)

    x_test_scaled = x_scaler.transform(x_test)



    y_scaler = MinMaxScaler()

    

    y_train_scaled = y_scaler.fit_transform(y_train)

    y_test_scaled = y_scaler.transform(y_test)

    

    return x_train_scaled,x_test_scaled,y_train_scaled,y_test_scaled
!ls ../input/
## Carregando dados

data = open_data()

data = remove_outliers(data)

data = imputation(data)
## Setando alvo

target_city = 'Monte Alegre'

target_names = ['TempBS_MTA','UmidR_MTA','Pres_MTA']        # Variaveis

shift_days = 3               # Numero de dias

shift_steps = shift_days * 3 # Número de horas

num_features = 9
df_targets = data[target_names].shift(-shift_days)
## Setando vetores

## Vetor entrada

x_data = data.values[0:-shift_steps]

## Vetor saída

y_data = df_targets.values[:-shift_steps]
### Teste Treinamento

# numero de dados de oberservação

num_data = len(x_data)



# fração para treinamento

train_split = 0.9



# número de elementos para treinamento/teste

num_train = int(train_split * num_data)

num_test = num_data - num_train
# vetores de entrada para teste e treinamento

x_train = x_data[0:num_train]

x_test = x_data[num_train:]



# vetores de saída para teste e treinamento

y_train = y_data[0:num_train]

y_test = y_data[num_train:]



# numero de atributos de entrada/saída

num_x_signals = x_data.shape[1]

num_y_signals = y_data.shape[1]
## Scaled

x_train_scaled,x_test_scaled,y_train_scaled,y_test_scaled = normalization(x_train,x_test,y_train,y_test)
#Get the data and splits in input X and output Y, by spliting in `n` past days as input X 

#and `m` coming days as Y.

def processData(data, look_back, forward_days,num_companies,jump=1):

    X,Y = [],[]

    for i in range(0,len(data) -look_back -forward_days +1, jump):

        X.append(data[i:(i+look_back)])

        Y.append(data[(i+look_back):(i+look_back+forward_days)])

    return np.array(X),np.array(Y)
X_test,y_test = processData(x_test_scaled,shift_steps,shift_days,num_features,shift_days)

y_test = np.array([list(a.ravel()) for a in y_test])



X,y = processData(x_train_scaled,shift_steps,shift_days,num_features)

y = np.array([list(x.ravel()) for x in y])



from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape)

print(X_validate.shape)

print(X_test.shape)

print(y_train.shape)

print(y_validate.shape)

print(y_test.shape)
NUM_NEURONS_FirstLayer = 200

NUM_NEURONS_SecondLayer = 100

EPOCHS = 250



#Build the model

model = Sequential()



model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(shift_steps,num_features), return_sequences=True))

model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,1)))

model.add(Dense(shift_days * num_features))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
path_checkpoint_LSTM = 'LSTM_single.keras'

callback_checkpoint_lstm = ModelCheckpoint(filepath=path_checkpoint_LSTM,

                                      monitor='val_loss',

                                      verbose=1,

                                      save_weights_only=True,

                                      save_best_only=True)



callback_tensorboard_lstm = TensorBoard(log_dir='../LSTM_log/',

                                   histogram_freq=0,

                                   write_graph=True,

                                   write_images=True)



callback_early_stopping_lstm = EarlyStopping(monitor='val_loss',

                                        patience=5, verbose=1)



callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',

                                       factor=0.1,

                                       min_lr=1e-4,

                                       patience=0,

                                       verbose=1)



callbacks_LSTM = [callback_early_stopping_lstm,

             callback_checkpoint_lstm,

             callback_tensorboard_lstm,

             callback_reduce_lr]
history = model.fit(X_train,y_train,

                    epochs=EPOCHS,

                    validation_data=(X_validate,y_validate),

                    shuffle=True,

                    batch_size=32,

                    callbacks=callbacks_LSTM,

                    verbose=1)
plt.figure(figsize=(15,5))

plt.title('Gráfico de perdas')

plt.plot(history.history['loss'], label='LSTM - loss',color='blue')

plt.plot(history.history['val_loss'], label='LSTM - val_loss',color='red')

plt.xlabel('Épocas')

plt.ylabel('%')

plt.legend()

plt.show()
import pandas as pd

dados_belterra = pd.read_csv("../input/dataset/dados_belterra.csv")

dados_monte_alegre = pd.read_csv("../input/dataset/dados_monte_alegre.csv")

dados_oriximina = pd.read_csv("../input/dataset/dados_oriximina.csv")