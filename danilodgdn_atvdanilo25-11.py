import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn import datasets as dt 

dados = pd.read_csv('/kaggle/input/hepatitis/hepatitis.csv')

dados.head()
dados.describe()
dados.shape
dados.info()
dados.isnull().sum()
import matplotlib.pyplot as plt

plt.hist(dados['age'], bins=10, rwidth=0.8)

plt.title("Histograma Idade")

plt.xlabel("Idade")

plt.ylabel("contagem")

plt.show()
X = dados.loc[:,'age':'histology']

y = dados['class']

print(X.shape,y.shape)
from sklearn.preprocessing import StandardScaler

SS = StandardScaler()

X = SS.fit_transform(X)
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(X_train,y_train)

y_predict = modelo.predict(X_test)
score = modelo.score(X,y)

score