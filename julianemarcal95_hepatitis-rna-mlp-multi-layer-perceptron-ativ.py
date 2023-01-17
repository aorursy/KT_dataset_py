import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dados = pd.read_csv("../input/hepatitis/hepatitis.csv")
dados.describe()

dados.shape
dados.info()
dados.isnull().sum()
import matplotlib.pyplot as plt 

plt.hist(dados['age'], bins=20, facecolor='red', alpha=0.5, rwidth=0.8) 

plt.show()
dados.keys()
X = dados.iloc[:,:19]

y = dados.histology
#X = dados.values

#Y = dados['histology']
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
score = modelo.score(X_test,y_test)

score