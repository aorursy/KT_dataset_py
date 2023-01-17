# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dflojas = pd.read_csv("../input/lojas.csv")

dflojas.head(10)
dftreino = pd.read_csv("../input/dataset_treino.csv")

dftreino.head(10)
dflojas.describe()
dfteste = pd.read_csv("../input/dataset_teste.csv")

dfteste.head()
dfteste.info()
plt.figure(figsize=(10, 12))

plt.scatter(dftreino['Customers'], dftreino['Sales'])
dftreino.info()
plt.figure(figsize=(10, 12))

plt.scatter(dftreino['Store'], dftreino['Sales'])
type(dftreino['Date'].iloc[0])
dftreino['Timestamp'] = pd.to_datetime(dftreino['Date'])
type(dftreino['Timestamp'].iloc[0])
dftreino.head()
sns.lmplot(x = 'Customers', y = 'Sales', data = dftreino)
plt.figure(figsize=(10, 12))

plt.hist(x = 'Customers', data = dftreino)
dftreino['Timestamp'].iloc[0].day
dftreino['Year'] = dftreino['Timestamp'].apply(lambda data: data.year)

dftreino['Month'] = dftreino['Timestamp'].apply(lambda data: data.month)

dftreino['Day'] = dftreino['Timestamp'].apply(lambda data: data.day)

dftreino.head()
porMes = dftreino.groupby('Month').count()

porMes['Sales'].head(12)
porMes['Sales'].plot()
dftreino.corr()
sns.pairplot(dftreino)
sns.distplot(dftreino['Sales'])
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(dftreino.corr(), annot = True, linewidths=.5, ax=ax)
dftreino.columns
dftreino.isnull().values.any()
dftreino[dftreino.isna().values]
dftreino.info()
dftreino.groupby('StateHoliday').count()
def strToInt(x):

    if x == 0:

        return 0

    if x == '0':

        return 1

    elif x == 'a':

        return 2

    elif x == 'b':

        return 3

    elif x == 'c':

        return 4

    

dftreino['StateHoliday'] = dftreino['StateHoliday'].apply(lambda x: strToInt(x))
dftreino.head()
dftreino.info()
X = dftreino[['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'Day']]

y = dftreino['Sales']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 64)
from sklearn.linear_model import LinearRegression
X_train.info()
lm = LinearRegression()

lm.fit(X_train, y_train)
#Verificando a qualidade do modelo

print(lm.intercept_)
lm.coef_
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeficiente'])

cdf
predicao = lm.predict(X_test)

predicao
plt.scatter(y_test, predicao)
fig, ax = plt.subplots(figsize=(10,10))

sns.distplot(y_test-predicao, label = 'Distribuição diferença entre o real e a predição', ax=ax)
dfteste.head()
dfteste['Timestamp'] = dfteste['Date'].apply(lambda x: pd.to_datetime(x))

dfteste['Year'] = dfteste['Timestamp'].apply(lambda time: time.year)

dfteste['Month'] = dfteste['Timestamp'].apply(lambda time: time.month)

dfteste['Day'] = dfteste['Timestamp'].apply(lambda time: time.day)

dfteste.head()
dfteste.columns
XTest = dfteste[['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'Day']]
predicaoDFTeste = lm.predict(X_test)

predicaoDFTeste
plt.scatter(y_test, predicaoDFTeste)
fig, ax = plt.subplots(figsize=(10,10))

sns.distplot(y_test-predicaoDFTeste, label = 'Distribuição diferença entre o real e a predição', ax=ax)
dfPredito = pd.DataFrame(predicaoDFTeste)

dfPredito.count()
dfPredito.describe()
dfPredito.to_csv('submission.csv', index = True)
from sklearn import metrics
metrics.mean_absolute_error(y_test, predicaoDFTeste)
metrics.mean_squared_error(y_test, predicaoDFTeste)
np.sqrt(metrics.mean_squared_error(y_test, predicaoDFTeste))