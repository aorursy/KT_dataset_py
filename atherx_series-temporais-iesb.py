# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings

warnings.filterwarnings("ignore")



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# lendo os dados

rawData = pd.read_excel('/kaggle/input/sries-temporais/Brasil.xlsx')

rawData.head()
# pegando apenas as observacoes a partir da linha 12

parseData = rawData.iloc[12::,:]
parseData.head().T
parseData = parseData.iloc[:,[0,1,4]]
parseData.head()
parseData.columns = ['mes', 'combustivel', 'precos']
parseData.info()
parseData.mes = pd.to_datetime(parseData.mes)

parseData.precos = pd.to_numeric(parseData.precos, downcast='float')
parseData.info()
parseData.head()
parseData = parseData[parseData.combustivel == 'ETANOL HIDRATADO']
parseData.info()
parseData.head()
rawData2 = pd.read_excel('/kaggle/input/sries-temporais/MENSAL_BRASIL-DESDE_Jan2013.xlsx')

rawData2.head()
# neste arquivo as observacoes comecao a partir da linha 15

parseData2 = rawData2.iloc[15:,:]
# pegando somente as colunas que nos interessa

parseData2 = parseData2.iloc[:,[0,1,4]]
# renomeando as colunas

parseData2.columns = ['mes', 'combustivel', 'precos']
parseData2.info()
# convertendo a coluna mes para datetime e o compo precos para numeric

parseData2.mes = pd.to_datetime(parseData2.mes)

parseData2.precos = pd.to_numeric(parseData2.precos, downcast='float')
parseData2.info()
# pegando apenas as observacoes referentes a ETANOL HIDRATADOeee

parseData2 = parseData2[parseData2.combustivel == 'ETANOL HIDRATADO']
parseData2.info()
parseData2.head()
data = pd.concat([parseData, parseData2])

data.sample(20), data.info()
plt.figure(figsize=(20,5))

plt.plot(data.mes, data.precos)

plt.grid(True)

plt.title('Evolucao preco Etanol')

plt.xlabel('preco')

plt.ylabel('ano')

plt.show()
data = data.iloc[:,[0,2]]

data.head()
# Como a regressao linear so trabalha com valores numericos nao sera possivel prever utilizando a data no formato date time, sera necessario converter para ordinal

X = data.mes.map(dt.datetime.toordinal).values; X[:5]
y = data.iloc[:,1].values; display(y[:5])

X = X.reshape(-1,1); X[:5] # foi necessario realizar um reshape na coluna para que pudessemos dar continuidade com a regressao linear
# separando o dado de treino de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# instanciando o modelo

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

display(lr.coef_[0])

display(lr.intercept_)
plt.figure(figsize=(20,5))



plt.plot(X, y)

plt.plot(X_test, lr_pred)

plt.show()
print(r2_score(y_test, lr_pred)); print(mean_squared_error(y_test, lr_pred))