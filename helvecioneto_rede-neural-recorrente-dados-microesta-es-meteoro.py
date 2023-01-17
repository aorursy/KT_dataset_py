%matplotlib inline

import warnings

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import datetime

from functools import reduce

from sklearn.preprocessing import MinMaxScaler



warnings.filterwarnings('ignore')
!pip install tensorflow==1.15.0rc3
# Bibliotecas Keras

import tensorflow as tf

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Input, Dense, LSTM, GRU, Embedding, Dropout

from tensorflow.python.keras.optimizers import RMSprop

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
print('Versão do TensorFlow: ',tf.__version__)

print('Versão do Keras: ',tf.keras.__version__)

print('Versão do Pandas: ',pd.__version__)
## Leitura Dados

# Monte Alegre

mtal = pd.read_csv('../input/dataset/dados_monte_alegre.csv', delimiter=';')

# Belterra

belt = pd.read_csv('../input/dataset/dados_belterra.csv', delimiter=';')

# Oriximina

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
print('Tamanho do dataframe Monte Alegre: ',mtal.shape[0], ', Variáveis:',mtal.shape[1])

print('Tamanho do dataframe Belterra: ',belt.shape[0], ', Variáveis:',belt.shape[1])

print('Tamanho do dataframe Obidos: ',orix.shape[0], ', Variáveis:',orix.shape[1])
## Merge entre os datasets

dfs = [mtal,belt,orix]

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
# Visualização do sensor Temperatura Bulbo Seco para as 3 estações

TempBulboS = dfs_final[['TempBS_MTA','TempBS_BLT','TempBS_OBD']].plot(figsize=(15,5),title='Temperatura de Bulbo Seco')

TempBulboS.legend(['Monte Alegre','Belterra','Óbidos'])

plt.xlabel('Data')

plt.ylabel('Temperatura em Cº')

plt.show()
# Visualização do sensor Umidade para as 3 estações

Humid = dfs_final[['UmidR_MTA','UmidR_BLT','UmidR_OBD']].plot(figsize=(15,5),title='Umidade Relativa do Ar')

Humid.legend(['Monte Alegre','Belterra','Óbidos'])

plt.xlabel('Data')

plt.ylabel('Umidade em %')

plt.show()
# Visualização do sensor Pressão Atmosférica para as 3 estações

PressaoP = dfs_final[['Pres_MTA','Pres_BLT','Pres_OBD']].plot(figsize=(15,5),title='Pressão Atmosférica')

PressaoP.legend(['Monte Alegre','Belterra','Óbidos'])

plt.xlabel('Data')

plt.ylabel('Pressão Atmosférica em hPa')

plt.show()
# Descrição do DataFrame

dfs_final.describe()
## Remoção de Outliers por quantil

low = .001

high = .9999

quant_df = dfs_final.quantile([low,high])
# Valores que representam os outliers

quant_df
# Função para correção dos outliers

filt_df = dfs_final.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 

                                    (x < quant_df.loc[high,x.name])], axis=0)



filt_df.describe()
# Numerode elementos sem registro no DataFrame sem correção

dfs_final.isna().sum()
# Numerode elementos sem registro no DataFrame com correção

filt_df.isna().sum()
# Visualização dos outliers por boxplot dados sem correção

import seaborn as sns
plt.figure(figsize=(15,3))

plt.title('Temperatura Bulbo Seco para estação Monte Alegre com outliers')

sns.boxplot(x=dfs_final['TempBS_MTA'])

plt.show()
# Visualização dos outliers por boxplot dados com correção

plt.figure(figsize=(15,3))

plt.title('Temperatura Bulbo Seco para estação Monte Alegre sem outliers')

sns.boxplot(x=filt_df['TempBS_MTA'])

plt.show()
# Visualização do sensor Pressão Atmosférica para as 3 estações

PressaoP = filt_df[['Pres_MTA','Pres_BLT','Pres_OBD']].plot(figsize=(15,5),title='Pressão Atmosférica')

PressaoP.legend(['Monte Alegre','Belterra','Óbidos'])

plt.xlabel('Data')

plt.ylabel('Pressão Atmosférica em hPa')

plt.show()
# Exemplo de visualização para imputação

# Exemplo entre os dias 01 e 07 de janeiro de 1996

preE = filt_df[['Pres_MTA','Pres_BLT','Pres_OBD']]['1996/01/01':'1996/01/07'].plot(figsize=(15,5),title='Pressão Atmosférica sem Imputação')

preE.legend(['Monte Alegre','Belterra','Óbidos'])

plt.xlabel('Data')

plt.ylabel('Pressão Atmosférica em hPa')

plt.show()
# Media entre Belterra e Oriximina

filt_df['Pres_MTA'] = filt_df.Pres_MTA.fillna(filt_df[['Pres_BLT','Pres_OBD']].mean(axis=1))
# Exemplo de visualização para imputação

# Exemplo entre os dias 01 e 07 de janeiro de 1996

preE = filt_df[['Pres_MTA','Pres_BLT','Pres_OBD']]['1996/01/01':'1996/01/07'].plot(figsize=(15,5),title='Pressão Atmosférica sem Imputação')

preE.legend(['Monte Alegre','Belterra','Óbidos'])

plt.xlabel('Data')

plt.ylabel('Pressão Atmosférica em hPa')

plt.show()
# Visualização do sensor Pressão Atmosférica para as 3 estações

PressaoP = filt_df[['Pres_MTA','Pres_BLT','Pres_OBD']].plot(figsize=(15,5),title='Pressão Atmosférica após imputação simples')

PressaoP.legend(['Monte Alegre','Belterra','Óbidos'])

plt.xlabel('Data')

plt.ylabel('Pressão Atmosférica em hPa')

plt.show()
# Numerode elementos sem registro no DataFrame com correção

filt_df.isna().sum()
# Exemplo de visualização para imputação

# Exemplo entre os dias 01 e 07 de janeiro de 1996

humE = filt_df[['UmidR_MTA','UmidR_BLT','UmidR_OBD']]['1995/12/27':'1996/01/07'].plot(figsize=(15,5),title='Umidade Relativa sem Imputação')

humE.legend(['Monte Alegre','Belterra','Óbidos'])

plt.xlabel('Data')

plt.ylabel('Umidade em %')

plt.show()
# Media entre Belterra e Oriximina

filt_df['UmidR_MTA'] = filt_df.UmidR_MTA.fillna(filt_df[['UmidR_BLT','UmidR_OBD']].mean(axis=1))
# Exemplo de visualização para imputação

# Exemplo entre os dias 01 e 07 de janeiro de 1996

humE = filt_df[['UmidR_MTA','UmidR_BLT','UmidR_OBD']]['1995/12/27':'1996/01/07'].plot(figsize=(15,5),title='Umidade Relativa com imputação estatística')

humE.legend(['Monte Alegre','Belterra','Óbidos'])

plt.xlabel('Data')

plt.ylabel('Umidade em %')

plt.show()
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)



## Imputação para Temperatura Bulbo Seco

timp = imp.fit(filt_df.loc[:,['TempBS_MTA','TempBS_BLT','TempBS_OBD']].to_numpy())

timp = timp.transform(filt_df.loc[:,['TempBS_MTA','TempBS_BLT','TempBS_OBD']].to_numpy())
df_final = filt_df.drop(['TempBS_MTA','TempBS_BLT','TempBS_OBD'], axis=1)
df_final['TempBS_MTA'] = timp[:,0]

df_final['TempBS_BLT'] = timp[:,1]

df_final['TempBS_OBD'] = timp[:,2]
df_final.isna().sum()
# Removendo os valores nulos

df_final = df_final.dropna()
df_final[['Pres_MTA','Pres_BLT','Pres_OBD']].plot(figsize=(15,5),title='Pressão Atmosférica após imputação simples')

plt.xlabel('Data')

plt.ylabel('Pressão Atmosférica em hPa')

plt.show()
df_final[['Pres_MTA','Pres_BLT','Pres_OBD']]['1988':'2013'].plot(figsize=(15,5),title='Pressão Atmosférica após imputação estatística final')

plt.xlabel('Data')

plt.ylabel('Pressão Atmosférica em hPa')

plt.show()
f, ax = plt.subplots(figsize=(15, 8))

plt.title('Matriz de Correlação entre as variáveis')

corr = df_final.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

plt.show()
df = df_final

df.head()
print('Quantidade de registros: ', df.values.shape[0], ', Quantidade de atributos: ', df.values.shape[1])
# df['Day'] = df.index.dayofyear

# df['Hour'] = df.index.hour
# Reorganizar dados

# df = df_final[['Day','Hour','TempBS_MTA','TempBU_MTA','UmidR_MTA','Pres_MTA',

#                'TempBS_BLT','TempBU_BLT','UmidR_BLT','Pres_BLT',

#                'TempBS_ORX','TempBU_ORX','UmidR_ORX','Pres_ORX']]



df = df_final[['TempBS_MTA','UmidR_MTA','Pres_MTA',

               'TempBS_BLT','UmidR_BLT','Pres_BLT',

               'TempBS_OBD','UmidR_OBD','Pres_OBD']]
df.head()
target_city = 'Monte Alegre'

target_names = ['TempBS_MTA','UmidR_MTA','Pres_MTA']

shift_days = 3

shift_steps = shift_days * 3 # Número de horas
# Novo dataframe

df_targets = df[target_names].shift(-shift_days)

df_targets
# Vetor de ENTRADA

x_data = df.values[0:-shift_steps]



print('Tipo dos dados: ',type(x_data))

print("Tamanho de entradas (observações) e número de atributos ", x_data.shape)
# Vetor de saída a partir do DataFrame target (Cidade Odense), ultimas linhas para previsão

y_data = df_targets.values[:-shift_steps]

print('Tipo dos dados: ',type(y_data))

print("Tamanho de saída (observações) e número de atributos ", y_data.shape)
print('Tamanho do Vetor de Entrada: ', len(x_data))

print('Tamanho do Vetor de Saída: ', len(y_data))
# numero de dados de oberservação

num_data = len(x_data)



# fração para treinamento

train_split = 0.9



# número de elementos para treinamento

num_train = int(train_split * num_data)

print('Número de elementos para treinamento: ', num_train)

num_test = num_data - num_train

print('Número de elementos para teste: ', num_test)
# vetores de entrada para teste e treinamento

x_train = x_data[0:num_train]

x_test = x_data[num_train:]
# vetores de saída para teste e treinamento

y_train = y_data[0:num_train]

y_test = y_data[num_train:]
# numero de atributos de entrada

num_x_signals = x_data.shape[1]

print('Atributos de entrada: ', num_x_signals)

num_y_signals = y_data.shape[1]

print('Atributos de saída: ', num_y_signals)
print("Min:", np.min(x_train))

print("Max:", np.max(x_train))
# objeto para os sinais de entrada

x_scaler = MinMaxScaler()
# função para normalização dos dados de entrada do treinamento

x_train_scaled = x_scaler.fit_transform(x_train)
print("Min:", np.min(x_train_scaled))

print("Max:", np.max(x_train_scaled))
# o mesmo objeto x_caler é utilizado para os valores de teste

x_test_scaled = x_scaler.transform(x_test)
# outra funcao para normalização dos dados de treinamento e teste da saída

y_scaler = MinMaxScaler()

y_train_scaled = y_scaler.fit_transform(y_train)

y_test_scaled = y_scaler.transform(y_test)
print("Min:", np.min(y_train_scaled))

print("Max:", np.max(y_train_scaled))
def batch_generator(batch_size, sequence_length):

    """

    Generator function for creating random batches of training-data.

    """



    # Infinite loop.

    while True:

        # Allocate a new array for the batch of input-signals.

        x_shape = (batch_size, sequence_length, num_x_signals)

        x_batch = np.zeros(shape=x_shape, dtype=np.float16)



        # Allocate a new array for the batch of output-signals.

        y_shape = (batch_size, sequence_length, num_y_signals)

        y_batch = np.zeros(shape=y_shape, dtype=np.float16)



        # Fill the batch with random sequences of data.

        for i in range(batch_size):

            # Get a random start-index.

            # This points somewhere into the training-data.

            idx = np.random.randint(num_train - sequence_length)

            

            # Copy the sequences of data starting at this index.

            x_batch[i] = x_train_scaled[idx:idx+sequence_length]

            y_batch[i] = y_train_scaled[idx:idx+sequence_length]

        

        yield (x_batch, y_batch)
# Tamanho do batch

batch_size = 19
# Sequência para cada batch

sequence_length = 1229
# gerador de dados em lote

generator = batch_generator(batch_size=batch_size,

                            sequence_length=sequence_length)
# teste para o gerador de dados em lote

x_batch, y_batch = next(generator)
x_train_scaled.shape
print(x_batch.shape)

print(y_batch.shape)
batch = 0   # Primeiro bloco 

signal = 2  #Sinal a partir dos dados com 9 atributos

seq = x_batch[batch, :, signal]

plt.figure(figsize=(15,3))

plt.title(label='Sinal de entrada com 9 atributos, tamanho do lote = 1229')

plt.plot(seq)

plt.show()
seq = y_batch[batch, :, signal]

plt.figure(figsize=(15,3))

plt.title(label='Sinal de saída com 9 atributos, tamanho do lote = 1229')

plt.plot(seq)

plt.show()
validation_data = (np.expand_dims(x_test_scaled, axis=0),

                   np.expand_dims(y_test_scaled, axis=0))
print('Formato dos dados de validação entrada: ', validation_data[0].shape )

print('Formato dos dados de validação saída: ', validation_data[1].shape )
# tensorflow e keras engine

model = Sequential()
print('Numéro de sinais de entrada:',num_x_signals)
model.add(GRU(units=256,

              return_sequences=True,

              input_shape=(None, num_x_signals,)))

model.add(Dropout(0.3,))
model_lstm = Sequential()

model_lstm.add(LSTM(units = 256,

                return_sequences=True,

                input_shape=(None,num_x_signals)))

model_lstm.add(Dropout(0.3))
#GRU Dense

model.add(Dense(num_y_signals, activation='sigmoid'))



#LSTM Dense

model_lstm.add(Dense(num_y_signals, activation='sigmoid'))
# periodos para warmup-period

warmup_steps = 50
def loss_mse_warmup(y_true, y_pred):

    

    """

    Calcula MSE entre as saídas verdadeiras e saídas previsas

    porém ignora a sequência inicial de aquecimento.

    

    y_true é a saída desejada.

    y_pred é a saída do modelo.

    """



    # tamanho para os dois tensores de entrada:

    #  Ignora parte da sequecência "warmup", tomando como medida parte do tamanho dos tensores



    y_true_slice = y_true[:, warmup_steps:, :]

    y_pred_slice = y_pred[:, warmup_steps:, :]



    # Calcula o MSE para cada valor de tensores

    loss = tf.losses.mean_squared_error(labels=y_true_slice,

                                        predictions=y_pred_slice)



    loss_mean = tf.reduce_mean(loss)



    return loss_mean
optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)

model.summary()
model_lstm.compile(loss=loss_mse_warmup, optimizer=optimizer)

model_lstm.summary()
path_checkpoint = 'GRU_256N_01.keras'

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,

                                      monitor='val_loss',

                                      verbose=1,

                                      save_weights_only=True,

                                      save_best_only=True)
path_checkpoint_LSTM = 'LSTM_256N_01.keras'

callback_checkpoint_lstm = ModelCheckpoint(filepath=path_checkpoint_LSTM,

                                      monitor='val_loss',

                                      verbose=1,

                                      save_weights_only=True,

                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',

                                        patience=5, verbose=1)
callback_tensorboard = TensorBoard(log_dir='./GRU_log/',

                                   histogram_freq=0,

                                   write_graph=True,

                                   write_images=True)
callback_tensorboard_lstm = TensorBoard(log_dir='./LSTM_log/',

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
callbacks = [callback_early_stopping,

             callback_checkpoint,

             callback_tensorboard,

             callback_reduce_lr]
callbacks_LSTM = [callback_early_stopping_lstm,

             callback_checkpoint_lstm,

             callback_tensorboard_lstm,

             callback_reduce_lr]
%%time

gru_history = model.fit_generator(generator=generator,

                    epochs=80,

                    steps_per_epoch=100,

                    validation_data=validation_data,

                    callbacks=callbacks)
%%time

lstm_history = model_lstm.fit_generator(generator=generator,

                    epochs=80,

                    steps_per_epoch=100,

                    validation_data=validation_data,

                    callbacks=callbacks_LSTM)
try:

    model.load_weights(path_checkpoint)

except Exception as error:

    print("Error trying to load checkpoint.")

    print(error)
plt.figure(figsize=(15,5))

plt.title('Gráfico de perdas')

plt.plot(lstm_history.history['loss'], label='LSTM - loss',color='green')

plt.plot(lstm_history.history['val_loss'], label='LSTM - val_loss',color='black')

plt.plot(gru_history.history['loss'], label='GRU - loss',color='blue')

plt.plot(gru_history.history['val_loss'], label='GRU - val_loss',color='red')

plt.xlabel('Épocas')

plt.ylabel('%')

plt.legend()

plt.show()
result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),

                        y=np.expand_dims(y_test_scaled, axis=0))
result2 = model_lstm.evaluate(x=np.expand_dims(x_test_scaled, axis=0),

                        y=np.expand_dims(y_test_scaled, axis=0))
print("loss (test-set):", result)
def output_frame_GRU(start_idx, length, train=True):

    """

    :param start_idx: Indice inicial da série temporal.

    :param length: Comprimento da sequência, número de elementos após o indice inicial.

    :param train: Valor Booleano para utilizar dados treinamento ou teste.

    """

    

    if train == True:

        # Usar dados de treinamento.

        x = x_train_scaled

        y_true = y_train

    else:

        # Usar dados de teste.

        x = x_test_scaled

        y_true = y_test

    

    # Indice final para sequência, tempo inicial mais comprimento.

    end_idx = start_idx + length

    

    #Selecione a seqüência do índice inicial especificado e comprimento.

    x = x[start_idx:end_idx]

    y_true = y_true[start_idx:end_idx]

    

    # Sinais de entrada para o modelo.

    x = np.expand_dims(x, axis=0)



    # Usar o modelo para prever os sinais de saída.

    y_pred = model.predict(x)

    

    # A saída do modelo tem valores entre 1 e 0.

    # Será necessário aplicar uma função de mapeamento inverso para deixar os dados rescalados

    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    

    output = pd.DataFrame()

    output['Prev_TempBS_MTA'] = y_pred_rescaled[:,0]

    output['Prev_UmidR_MTA'] = y_pred_rescaled[:,1]

    output['Prev_Pres_MTA'] = y_pred_rescaled[:,2]

    output['Datetime'] = df[0:-shift_steps][num_train:].index[start_idx:start_idx+length]

    output = output.set_index('Datetime')

    

    return output
def output_frame_LSTM(start_idx, length, train=True):

    """

    :param start_idx: Indice inicial da série temporal.

    :param length: Comprimento da sequência, número de elementos após o indice inicial.

    :param train: Valor Booleano para utilizar dados treinamento ou teste.

    """

    

    if train == True:

        # Usar dados de treinamento.

        x = x_train_scaled

        y_true = y_train

    else:

        # Usar dados de teste.

        x = x_test_scaled

        y_true = y_test

    

    # Indice final para sequência, tempo inicial mais comprimento.

    end_idx = start_idx + length

    

    #Selecione a seqüência do índice inicial especificado e comprimento.

    x = x[start_idx:end_idx]

    y_true = y_true[start_idx:end_idx]

    

    # Sinais de entrada para o modelo.

    x = np.expand_dims(x, axis=0)



    # Usar o modelo para prever os sinais de saída.

    y_pred = model_lstm.predict(x)

    

    # A saída do modelo tem valores entre 1 e 0.

    # Será necessário aplicar uma função de mapeamento inverso para deixar os dados rescalados

    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    

    output = pd.DataFrame()

    output['Prev_TempBS_MTA'] = y_pred_rescaled[:,0]

    output['Prev_UmidR_MTA'] = y_pred_rescaled[:,1]

    output['Prev_Pres_MTA'] = y_pred_rescaled[:,2]

    

    output['Datetime'] = df[0:-shift_steps][num_train:].index[start_idx:start_idx+length]

    output = output.set_index('Datetime')

    

    return output
length = 500

start_idx = 0
# apenas com os dados de treinamento

saida_LSTM = output_frame_LSTM(start_idx=start_idx, length=length, train=False)

saida_GRU = output_frame_GRU(start_idx=start_idx, length=length, train=False)

saida_LSTM.head()
from math import sqrt

from sklearn.metrics import mean_squared_error
x = df['TempBS_MTA'][0:-shift_steps][num_train:][start_idx:start_idx+length].values.reshape(-1,1)

y_L = saida_LSTM['Prev_TempBS_MTA'].values.reshape(-1,1)

y_G = saida_GRU['Prev_TempBS_MTA'].values.reshape(-1,1)



xscaler = MinMaxScaler()



scalx = xscaler.fit_transform(x)

scaly_L = xscaler.fit_transform(y_L)

scaly_G = xscaler.fit_transform(y_G)



plt.figure(figsize=(15,5))

plt.plot(df['TempBS_MTA'][0:-shift_steps][num_train:][start_idx:start_idx+length], label='Verdadeira',)

plt.plot(saida_LSTM['Prev_TempBS_MTA'], label='Prevista')

rmse = sqrt(mean_squared_error(scalx,scaly_L))*100

plt.title('Temperatura de Bulbo Seco - LSTM: atual vs previsto: error: %.3f' %rmse + '%')

plt.legend()

plt.show()
plt.figure(figsize=(15,5))

plt.plot(df['TempBS_MTA'][0:-shift_steps][num_train:][start_idx:start_idx+length], label='Verdadeira')

plt.plot(saida_GRU['Prev_TempBS_MTA'], label='Prevista')

rmse = sqrt(mean_squared_error(scalx,scaly_G))*100

plt.title('Temperatura de Bulbo Seco - GRU: atual vs previsto: error: %.3f' %rmse + '%')

plt.legend()

plt.show()
plt.figure(figsize=(15,5))

plt.plot(df['UmidR_MTA'][0:-shift_steps][num_train:][start_idx:start_idx+length][start_idx:start_idx+length], label='Verdadeira')

plt.plot(saida_GRU['Prev_UmidR_MTA'], label='Prevista')

plt.legend()
plt.figure(figsize=(15,5))

plt.plot(df['Pres_MTA'][0:-shift_steps][num_train:][start_idx:start_idx+length][start_idx:start_idx+length], label='Verdadeira')

plt.plot(saida_GRU['Prev_Pres_MTA'], label='Prevista')

plt.legend()
def plot_comparison(start_idx, length=100, train=True):

    """

    Plot the predicted and true output-signals.

    

    :param start_idx: Start-index for the time-series.

    :param length: Sequence-length to process and plot.

    :param train: Boolean whether to use training- or test-set.

    """

    

    if train:

        print('s')

        # Use training-data.

        x = x_train_scaled

        y_true = y_train

    else:

        # Use test-data.

        x = x_test_scaled

        y_true = y_test

    

    # End-index for the sequences.

    end_idx = start_idx + length

    

    # Select the sequences from the given start-index and

    # of the given length.

    x = x[start_idx:end_idx]

    y_true = y_true[start_idx:end_idx]

    

    # Input-signals for the model.

    x = np.expand_dims(x, axis=0)



    # Use the model to predict the output-signals.

    y_pred = model_lstm.predict(x)

    

    # The output of the model is between 0 and 1.

    # Do an inverse map to get it back to the scale

    # of the original data-set.

    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    

    # For each output-signal.

    for signal in range(len(target_names)):

        # Get the output-signal predicted by the model.

        signal_pred = y_pred_rescaled[:, signal]

        

        # Get the true output-signal from the data-set.

        signal_true = y_true[:, signal]



        # Make the plotting-canvas bigger.

        plt.figure(figsize=(15,5))

        

        # Plot and compare the two signals.

        plt.plot(signal_true, label='true')

        plt.plot(signal_pred, label='pred')

        

        # Plot grey box for warmup-period.

        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        

        # Plot labels etc.

        plt.ylabel(target_names[signal])

        plt.legend()

        plt.show()
plot_comparison(start_idx=200, length=500, train=False)
def plot_comparison_GRU(start_idx, length=100, train=True):

    """

    Plot the predicted and true output-signals.

    

    :param start_idx: Start-index for the time-series.

    :param length: Sequence-length to process and plot.

    :param train: Boolean whether to use training- or test-set.

    """

    

    if train:

        print('s')

        # Use training-data.

        x = x_train_scaled

        y_true = y_train

    else:

        # Use test-data.

        x = x_test_scaled

        y_true = y_test

    

    # End-index for the sequences.

    end_idx = start_idx + length

    

    # Select the sequences from the given start-index and

    # of the given length.

    x = x[start_idx:end_idx]

    y_true = y_true[start_idx:end_idx]

    

    # Input-signals for the model.

    x = np.expand_dims(x, axis=0)



    # Use the model to predict the output-signals.

    y_pred = model.predict(x)

    

    # The output of the model is between 0 and 1.

    # Do an inverse map to get it back to the scale

    # of the original data-set.

    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    

    # For each output-signal.

    for signal in range(len(target_names)):

        # Get the output-signal predicted by the model.

        signal_pred = y_pred_rescaled[:, signal]

        

        # Get the true output-signal from the data-set.

        signal_true = y_true[:, signal]



        # Make the plotting-canvas bigger.

        plt.figure(figsize=(15,5))

        

        # Plot and compare the two signals.

        plt.plot(signal_true, label='true')

        plt.plot(signal_pred, label='pred')

        

        # Plot grey box for warmup-period.

        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        

        # Plot labels etc.

        plt.ylabel(target_names[signal])

        plt.legend()

        plt.show()
plot_comparison_GRU(start_idx=200, length=500, train=False)