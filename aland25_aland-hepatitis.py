import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dados = pd.read_csv("/kaggle/input/hepatitis/hepatitis.csv") #importa dataset
dados.head() #mostra dataset
dados.describe() #ex1 - 1 matriz dataset
dados.shape #ex2 - 120colunas e 20 linhas
dados.info() #ex3 - mostras os tipos dos dados ex inteiro ou float
dados.isnull().sum() #4 mostra quantidade demtro das colunas os campos vazios NO CASO NAO HA se houvesse usaria comando dropna
import matplotlib.pyplot as plt # importa biblioteca grafica

plt.figure(figsize=(10,5))#tamanho do grafico delimitando

plt.hist(dados['age'],bins=10,rwidth=0.9,facecolor='black', alpha=0.8,) #espaçocolunas0,9 espessura15 corazul e transparencia0,5 

plt.axis([0,80,0,40])#define o tamanho do grafico x a Y eixos do grafico de acordo com scribe idade maxima 78 entao 80 limite

plt.title("GRAFICO DE IDADE",  color='red')

plt.xlabel("idades",  color='white')#idade de 0 80

plt.ylabel("Numero de pessoas",  color='white')

plt.grid()# grade

plt.show()
X = dados.iloc[:,1:]

y = dados.loc[:,'class']

#X.head()

y

#y.head()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)

X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#X_train

#X_test 

#y_train 

#y_test 