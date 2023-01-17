import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dados = pd.read_csv('/kaggle/input/hepatitis/hepatitis.csv')
dados.describe()
dados.shape
dados.info()
dados.isnull().sum()
import matplotlib.pyplot as plt



plt.hist(dados['age'], bins = 8, rwidth = 0.5)

plt.title('Histograma Idade')

plt.xlabel('idade')

plt.ylabel('contagem')

plt.show()
X = dados.iloc[:,1:]

y = dados['class']
print(X.keys())
X.head()
y.head()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(X_train, y_train)

y_predict = modelo.predict(X_test)

score = modelo.score(X,y)

score
score = modelo.score(X,y)

score