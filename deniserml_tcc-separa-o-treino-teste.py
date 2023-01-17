# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importando as Bibliotecas

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Importando o arquivo

#Importing File

file = "../input/COTAHIST_A2009_to_A2018P.csv"

df = pd.read_csv(file)

df.head(2)
#Eliminando o indice não utilizado

df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

df.head(2)
#Descobrindo os códigos BDI do arquivo

codBDI = df[['CODBDI']]

codBDI = np.unique(codBDI)

codBDI
#Valores de Especificação de Papel existentes no arquivo

codESPECI = df [['ESPECI']]

codESPECI = np.unique(codESPECI)

codESPECI
#Tipos de Mercado existentes no arquivo

tpMercado = df [['TPMERC']]

tpMercado = np.unique(tpMercado)

tpMercado
'''

Eliminando os mercados que não serão utilizados em nossa análise.Esses mercados são: LEILÃO (017), FRACIONARIO(020) 

e o TERMO(030)

'''

mask = ((df['TPMERC'] == 10) | (df['TPMERC'] == 12) | (df['TPMERC'] == 13) | (df['TPMERC'] == 70) | (df['TPMERC'] == 80))

new_df = df[mask]

new_df.head(2)
#Moeda utilizada no pregão

moedaMerc = new_df [['MODREF']]

moedaMerc = np.unique(moedaMerc)

moedaMerc
#Taxa de correção dos contratos nos arquivos.

indCorr = new_df [['INDOPC']]

indCorr = np.unique(indCorr)

indCorr
#FATCOT - FATOR DE COTAÇÃO DO PAPEL

fatCot = new_df [['FATCOT']]

fatCot = np.unique(fatCot)

fatCot
'''

Eliminando a coluna TIPREG, pois esta possui um valor fixo que não será utilizado na análise

'''

new_df.drop(columns=['TIPREG'], axis = 1, inplace = True)

new_df.head(2)
'''

Eliminando a coluna PRAZOT, pois este campo é referente ao prazo do hedge do mercado a termo.Essa coluna possui valores

nulos para os mercados de opções e a vista. Como esta não fará parte da analise, pode ser eliminada.

'''

new_df.drop(columns=['PRAZOT'], axis = 1, inplace = True)

new_df.head(10)
'''

Será que devemos retirar?

Dúvidas referente aos campos: CODBDI - CÓDIGO BDI , ESPECI - ESPECIFICAÇÃO DO PAPEL, CODISI e DISMES

'''



'''

O que faremos com a data de vencimento do mercado a vista?

'''
#new_df = new_df[['DATPRE','CODNEG','TPMERC','PREABE','PREMAX','PREMIN','PREMED','PREULT','PREOFC','PREOFV','PREEXE','DATVEN']]

#new_df.head(2)



new_df = new_df[['TPMERC','PREABE','PREMAX','PREMIN','PREMED','PREULT','PREOFC','PREOFV','PREEXE']]

new_df.head(2)
X = new_df.drop(['PREULT'],axis=1)

y = new_df.PREULT
print(X.shape)

print(y.shape)
# separando os dados em treino e teste

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)



# separando o conjunto de treino em validação também

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)
from sklearn import model_selection

from sklearn import linear_model



model = linear_model.LinearRegression()

model.fit(X_train, y_train)

model.score(X_test, y_test)
from sklearn import metrics



erro_treino = metrics.mean_squared_error(y_train,model.predict(X_train))

print('RMSE no treino:', erro_treino)



erro_teste = metrics.mean_squared_error(y_test,model.predict(X_test))

print('RMSE no teste:', erro_teste)
from sklearn.model_selection import cross_val_score

resultado = cross_val_score(model, X_test, y_test, cv = 10)

print(resultado.mean())
import keras as K

import tensorflow as tf

import pandas as pd

import seaborn as sns

import os

from matplotlib import pyplot as plt

%matplotlib inline

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#Montando a rede neural

tf.logging.set_verbosity(tf.logging.ERROR) #desliga os warnings do tensorflow



#Inicializador

init = K.initializers.RandomUniform(seed=1)



#Criando o otimizador

simple_sgd = K.optimizers.SGD(lr=0.010)



#Construindo o modelo (topologia)

model = K.models.Sequential()

model.add(K.layers.Dense(units=10, input_dim=8, kernel_initializer=init, activation='tanh')) #1a camada oculta

model.add(K.layers.Dense(units=10, kernel_initializer=init, activation='tanh'))               #2a camada oculta

model.add(K.layers.Dense(units=1,  kernel_initializer=init, activation=None))               #Camada de saída



#Compilando o modelo

model.compile(loss='mean_squared_error', optimizer=simple_sgd, metrics=['mse'])
#Treinamento

batch_size = 8

max_epochs = 2

print("Iniciando treinamento... ")

h = model.fit(X_train, y_train, batch_size=batch_size, epochs=max_epochs, shuffle=True, verbose=1)

print("Treinamento finalizado \n")
#Treinamento

y_pred = model.predict(X_train)

y_d = np.array(y_train).reshape(-1, 1)



results = abs(y_pred - y_d) < np.abs(0.15 * y_d)

results



#Acuracidade

acc = np.sum(results) / len(results)

print("Taxa de acerto do conjunto de treinamento (%): {0:.4f}".format(acc*100) )



#Teste

y_pred = model.predict(X_test)

y_d = np.array(y_test).reshape(-1, 1)



results = abs(y_pred - y_d) < np.abs(0.15 * y_d)

results



#Acuracidade

acc = np.sum(results) / len(results)

print("Taxa de acerto do conjunto de teste (%): {0:.4f}".format(acc*100) )
# 5 Avaliação do modelo

eval = model.evaluate(X_train, y_train, verbose=0)

print("Erro médio do conjunto de treinamento {0:.4f}".format(eval[0]))



eval = model.evaluate(X_test, y_test, verbose=0)

print("Erro médio do conjunto de teste {0:.4f}".format(eval[0]))
# 7. Usando modelo (operação)

np.set_printoptions(precision=4)

unknown = np.full(shape=(1,8), fill_value=0.6, dtype=np.float32)

unknown[0][3] = -1.0 # encodando o booleano

predicted = model.predict(unknown)

print("Usando o modelo para previsão de preço médio de casa para as caracteristicas: ")

print(unknown)

print("\nO Valor da Opção Amanhã será [REAIS]: ")

print(predicted)