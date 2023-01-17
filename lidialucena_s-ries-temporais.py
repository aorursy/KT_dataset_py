# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Primeiramente, irei baixar as bibliotecas necessárias para o presente estudo



import pandas as pd 

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split

from statsmodels.tsa.holtwinters import ExponentialSmoothing
#Agora, importarei o arquivo do presente estudo



Brasil = pd.read_excel('/kaggle/input/sries-temporais/Brasil.xlsx')

Brasil.head()

#Retirando o head da tabela

parsedData= Brasil.iloc[12:,:] 
# Selecionando algumas colunas

parsedData = parsedData.iloc[:,[0,1,4]] 
# Alterando os nomes das colunas

parsedData.columns = ['mes', 'combustiveis','precos'] 
# Identificando os tipos de variáveis

parsedData.info()
# Identificando os primeiros registros da base

parsedData.head()
#Alterando o formato do mês e do preço

parsedData['mes'] = pd.to_datetime(parsedData['mes'])

parsedData['precos'] = pd.to_numeric(parsedData['precos'], downcast='float')
# Detalha o novo formato das variáveis mês e preço

parsedData.info()
# Traz a nova base formatada

parsedData.head()

# Selecionando o tipo de combustível 'Etanol Hidratado'

parsedData = parsedData[parsedData['combustiveis'] == 'ETANOL HIDRATADO']
# Seleciona as colunas que informam o mês e os preços 

parsedData = parsedData.iloc[:,[0,2]]
# Agora, vou baixar o arquivo que trata das informações a partir de janeiro de 2013

Brasil2013 = pd.read_excel('/kaggle/input/sries-temporais/MENSAL_BRASIL-DESDE_Jan2013.xlsx')

Brasil2013.head()

#Limpando a base



parsedData2 = Brasil2013.iloc[15:,:] 
# Selecionando algumas colunas

parsedData2 = parsedData2.iloc[:,[0,1,4]] 
# Alterando os nomes das colunas

parsedData2.columns = ['mes', 'combustiveis', 'precos'] 
#Alterando o formato do mês e do preço

parsedData2['mes'] = pd.to_datetime(parsedData2['mes'])

parsedData2['precos'] = pd.to_numeric(parsedData2['precos'], downcast='float')
# Selecionando o tipo de combustível 'Etanol Hidratado'

parsedData2 = parsedData2[parsedData2['combustiveis'] == 'ETANOL HIDRATADO']

# Seleciona as colunas que informam o mês e os preços 

parsedData2 = parsedData2.iloc[:,[0,2]]
#Agora irei juntar os dois dataframes

Brasil_combustivel = pd.concat([parsedData, parsedData2])

Brasil_combustivel.tail()

Brasil_combustivel.info()
# Os dados em gráfico



plt.figure(figsize=(20,5)) 

plt.plot(Brasil_combustivel['mes'], Brasil_combustivel['precos'])
X = Brasil_combustivel.iloc[:,0].map(dt.datetime.toordinal).values 
y = Brasil_combustivel.iloc[:,1].values

X = X.reshape(-1,1) 

lr = LinearRegression() 

lr.fit(X, y) 

display(lr.coef_) 

display(lr.intercept_) 
plt.figure(figsize=(20,5)) 

plt.plot(X, lr.predict(X))

plt.plot(Brasil_combustivel['mes'], Brasil_combustivel['precos'])
#Série temporal - Naive

y = np.asarray(Brasil_combustivel['precos'], dtype='int')

nb_b = BernoulliNB()

nb_m = MultinomialNB()

nb_g = GaussianNB()

nb_b.fit(X,y)

nb_m.fit(X,y)

nb_g.fit(X,y)
# Mostra eixo Y

y
# mostra eito X

X

#Série temporal - projeção



plt.figure(figsize=(20,5)) 

plt.plot(Brasil_combustivel['mes'], Brasil_combustivel['precos'])

plt.plot(X, nb_g.predict(X))

plt.plot(X, nb_b.predict(X))

plt.plot(X, nb_m.predict(X))