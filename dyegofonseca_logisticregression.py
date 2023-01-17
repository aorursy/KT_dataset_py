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
df = pd.read_csv('../input/advertising.csv')
df.head()
df.info()
df.describe()
# Verificando se há dados em branco

sns.heatmap(df.isnull())
plt.figure(figsize=(15,8))

sns.heatmap(df.corr(), annot=True,)
plt.figure(figsize=(15,8))

sns.distplot(df['Age'],bins=60)

plt.xlabel('Age')
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=df, kind='kde')
df.columns
sns.jointplot(x='Clicked on Ad', y='Daily Time Spent on Site', data=df, kind='kde')

sns.jointplot(x='Clicked on Ad', y='Daily Internet Usage', data=df, kind='kde', color='red')
sns.pairplot(df)
from sklearn.model_selection import train_test_split
X = df[['Daily Time Spent on Site', 

        'Age', 

        'Area Income',

        'Daily Internet Usage',

        'Male',

    ]]

y = df['Clicked on Ad']
# Segmentando os dados de treino e de teste

Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.30)
# Penas verificando o tamanho das bases de treino e teste

print('Tamanho do Xtrain: ', Xtrain.shape,

      '\nTamanho do Xtest: ', Xtest.shape, 

      '\nTamanho do yTrain: ', yTrain.shape, 

      '\nTamanho do yTest ', yTest.shape)
# Importando o modelo de Regressão Linear

from sklearn.linear_model import LogisticRegression
# Treinando o modelo de Regressão Logística

logModel = LogisticRegression()

logModel.fit(Xtrain, yTrain)
# Realizando a previsão utilizando como base o Xtest

predict = logModel.predict(Xtest)
# Criando o relatório de classifição do modelo

from sklearn.metrics import classification_report
print(classification_report(yTest, predict))
# Com base nas informações que foram informadas no banco de dados podemos afirmar que o modelo teve precisão de 90%, ou seja,

# com esse modelo podemos prever com 90% se haverá ou não um evento de click do usuário