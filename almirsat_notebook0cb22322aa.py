  #Bibliotecas para manipulação dos dados

import pandas as pd

import numpy as np

from datetime import datetime

    

    #Bibliotecas de gráfico

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=20,10

plt.style.use('ggplot')

    

    #Bibliotecas para construímos a LSTM

from keras.models import Sequential

from keras.layers import LSTM,Dropout,Dense,LeakyReLU

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

    

    #Biblioteca para retirar avisos

import warnings

warnings.filterwarnings("ignore")
#Leitura dos dados

dados=pd.read_csv("../input/apple-stockscsv/Apple_stocks.csv",sep=",")
#Conhecendo os dados que iremos trabalhar

dados.head()
dados.dtypes
#Vamos criar uma cópia dos dados originais para podermos manipular os dados.

df = dados.copy()
#Renomeando as colunas

df.columns = ["Date","Close","Volume","Open","High","Low"]
#Definindo a função para alteramos as datas

def to_string_date(x:str):

    return pd.to_datetime(

                datetime.strftime(

                      datetime.strptime(x.replace("/","-"),"%m-%d-%Y"),"%Y-%m-%d"))
#Definindo a função para alteramos o valor de fechamento

def to_float(x:str):

    return float(x[x.find("$")+1:])
#Aplicamos as funções nas colunas que precisam de transformação

df["Date"] = df["Date"].apply(to_string_date)

df["Close"] = df["Close"].apply(to_float)

df = df.sort_values('Date')
#Visualização dos dados após alteração

df.head()
df.index = df["Date"]

df = df["Close"]
#Vamos analisar o comportamento da nossa variável resposta

plt.figure(figsize=(16,8))

plt.ylabel("Preço de Fechamento em USD($)",fontsize=15)

plt.title('Série de Preço de Fechamento',fontsize=25)

plt.xticks(rotation= 45,fontsize=15)

plt.plot(df)
dataset = df.values.reshape((-1,1))

scaler = MinMaxScaler(feature_range=(0, 1)) 

dataset = scaler.fit_transform(dataset)
'''Para treinar nossa rede, usaremos um intervalo de tempo para prevermos o nosso próximo valor'''

look_back = 5

future_target = 1
'''Separando os dados em treino e teste.'''

tam = int(len(dataset) * 0.70)

dataset_teste = dataset[tam:]

dataset_treino = dataset[:tam]
#Função de transformação dos dados

def process_data(data, look_back, forward_days,jump=1):

    X,Y = [],[]

    for i in range(0,len(data) -look_back -forward_days +1, jump):

        X.append(data[i:(i+look_back)])

        Y.append(data[(i+look_back):(i+look_back+forward_days)])

    return np.array(X),np.array(Y)

X, y = process_data(dataset_treino,look_back,future_target)

y = np.array([list(a.ravel()) for a in y])



x_test, y_test = process_data(dataset_teste,look_back,future_target)

y_test = np.array([list(a.ravel()) for a in y_test])
X[0]
y[0]
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
#Definindo os números de neurônios por camada

n_first = 128

EPOCHS = 50

#Construido o modelo

model = Sequential()

model.add(LSTM(n_first,input_shape = (look_back,1)))

model.add(LeakyReLU(alpha=0.3))

model.add(Dropout(0.3))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')



history = model.fit(X_train,y_train,epochs=EPOCHS,validation_data=(X_validate,y_validate),shuffle=False,batch_size=2, verbose=2)
#Gráfico do resultado da função perda por epochs

plt.figure(figsize = (15,10))

plt.plot(history.history['loss'], label='loss')

plt.plot(history.history['val_loss'], label='val_loss')

plt.ylabel("Número de Epochs")

plt.legend(loc='best',fontsize=15)

plt.show()
#Salvando os valores preditos

Xt = model.predict(x_test)
plt.figure(figsize = (15,10))

plt.plot(scaler.inverse_transform(y_test.reshape(-1,1)),c='b', label='Teste')

plt.plot(scaler.inverse_transform(Xt.reshape(-1,1)), c='r',label='Predito')

plt.ylabel("Preço de Fechamento em USD($)")

plt.legend(loc='best')

plt.show()