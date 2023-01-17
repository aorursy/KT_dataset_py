import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

hepatitis = pd.read_csv("/kaggle/input/hepatitis/hepatitis.csv")

hepatitis.head()
hepatitis.describe()
hepatitis.shape
hepatitis.info()
hepatitis.isnull().sum()
import matplotlib.pyplot as plt

plt.hist(hepatitis['age'], bins=10, rwidth=0.8)

plt.title("Distribuição de Idades")

plt.xlabel("Idade")

plt.ylabel("Quantidade com Hepatitis")

plt.grid()



plt.show()
hepatitis.keys()

X = hepatitis.iloc[:,1:]

y = hepatitis.iloc[:,:1]
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(X_train, y_train)

y_predict = modelo.predict(X_test)
score = modelo.score(X_test, y_test)

score