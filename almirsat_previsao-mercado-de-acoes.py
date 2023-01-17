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

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

    

    #Bibliotecas para construímos a LSTM

from keras.models import Sequential

from keras.layers import LSTM,Dropout,Dense,LeakyReLU

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

    

    #Biblioteca para retirar avisos

import warnings

warnings.filterwarnings("ignore")
#carregando a base histórica da BBAS3

base=pd.read_excel('../input/cotacaobbas3/cotacao.xls',header=1, decimal=',')
#invertendo dataset para ordem crescente de datas

base=base.loc[::-1]
#visualizando as primeiras linhas do dataset

base.head()
base.tail()
#verificação de existência de dados nulos

base.isnull().sum()
#descrição dos dados

pd.set_option('display.float_format','{:.2f}'.format)

base.describe().transpose()
#numero de linhas e colunas do dataset

base.shape
#indexação do dataset usando a coluna Data

base.index = base["Data"]
plt.figure(figsize=(10,7))

sns.set_context('notebook', font_scale=1.5, rc={'font.size':20,

                                               'axes.titlesize':20,

                                               'axes.labelsize':18})

sns.kdeplot(base['Abertura'],color='green')

sns.rugplot(base['Abertura'],color='red')

sns.distplot(base['Abertura'],color='green')

sns.set_style('darkgrid')



plt.xlabel('Distribuicao do Preco de Abertura');
trace1 = go.Scatter(x = base.Data,

                    y = base.Fechado,

                    mode = 'lines',

                    name = 'Fechado',

                    line = {'color': 'Blue'})



trace2 = go.Scatter(x = base.Data,

                    y = base.Abertura,

                    mode = 'lines',

                    name = 'Abertura',

                    line = {'color': 'red'})

layout = go.Layout(title='Histórico de Preço da BBAS3 de 04/2015 até 04/2020',

                   yaxis={'title':'Preço R$'},

                   xaxis={'title': 'Data'})

data = [trace1, trace2]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
#Vamos criar uma cópia dos dados originais para podermos manipular os dados.

df = base.copy()
df.index = df["Data"]

df = df["Fechado"]
#processo para normalizar os dados

dataset = df.values.reshape((-1,1))

scaler = MinMaxScaler(feature_range=(0, 1)) 

dataset = scaler.fit_transform(dataset)
#'''Para treinar nossa rede, usaremos um intervalo de tempo para prevermos o nosso próximo valor'''

look_back = 30

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
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
#Definindo os números de neurônios por camada

n_first = 128

EPOCHS = 50

#Construido o modelo

model = Sequential()

model.add(LSTM(n_first,input_shape = (look_back,1)))

model.add(LeakyReLU(alpha=0.3))

model.add(Dropout(0.3))

model.add(Dense(future_target))

model.compile(loss='mean_squared_error', optimizer='adam')



history = model.fit(X_train,y_train,epochs=EPOCHS,validation_data=(X_validate,y_validate),shuffle=False,batch_size=2, verbose=1)
#Gráfico do resultado da função perda por epochs

plt.figure(figsize = (15,10))

plt.plot(history.history['loss'], label='loss')

plt.plot(history.history['val_loss'], label='val_loss')

plt.ylabel("Número de Epochs")

plt.legend(loc='best',fontsize=15)

plt.show()
#Salvando os valores preditos

Xt = model.predict(x_test)
Xt.shape
teste= pd.DataFrame((scaler.inverse_transform(y_test.reshape(-1,1))))[0]

predito=pd.DataFrame(scaler.inverse_transform(Xt.reshape(-1,1)))[0]
from sklearn.metrics import mean_squared_error

from math import sqrt



rmse = sqrt(mean_squared_error(teste, predito))
rmse
# com 30 épocas o rms foi de 1.7830049807182384
# com 50 épocas o rms foi de 1.2940400177566898
tam-look_back
#ultimo modelo de gráfico ajustado ao look_back

trace1 = go.Scatter(x = df.index[tam+look_back :],

                    y = base.Abertura[tam+look_back :],

                    mode = 'lines',

                    name = 'Teste',

                    line = {'color': 'Blue'})



trace2 = go.Scatter(x = df.index[tam+look_back:],

                    y = predito[1:],

                    mode = 'lines',

                    name = 'Predito',

                    line = {'color': 'red'})

layout = go.Layout(title='comparativo entre teste e predito com 50 épocas',

                   yaxis={'title':'Preço R$'},

                   xaxis={'title': 'Data'})

data = [trace1, trace2]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
#treinamento com 30 épocas

plt.figure(figsize = (15,10))

plt.plot((scaler.inverse_transform(y_test.reshape(-1,1))-scaler.inverse_transform(Xt.reshape(-1,1)))/scaler.inverse_transform(Xt.reshape(-1,1)))

#plt.plot(scaler.inverse_transform(Xt.reshape(-1,1)), c='r',label='Predito')

plt.ylabel("diferença entre valor de teste e valor predito")

plt.show()
#treinamento com 30 épocas

plt.figure(figsize = (15,10))

plt.plot((scaler.inverse_transform(y_test.reshape(-1,1))-scaler.inverse_transform(Xt.reshape(-1,1))))#/scaler.inverse_transform(Xt.reshape(-1,1)))

#plt.plot(scaler.inverse_transform(Xt.reshape(-1,1)), c='r',label='Predito')

plt.ylabel("diferença entre valor de teste e valor predito")

plt.xlabel("50 épocas")

plt.show()