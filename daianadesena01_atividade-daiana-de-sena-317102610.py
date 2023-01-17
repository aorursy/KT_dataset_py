import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv ("/kaggle/input/hepatitis/hepatitis.csv")
df.describe ()

df.shape
df.isnull().sum()
df.info()
import matplotlib.pyplot as pl



pl.hist(df['age'], bins=None,rwidth=0.5)

pl.title('Histograma Idade.')

pl.xlabel('idade')

pl.ylabel('contagem')

pl.show()
X = df.iloc[:,1:]

Y = df['class']

X.head()
Y.head()
from sklearn.preprocessing import StandardScaler #as ss

ss = StandardScaler()

X = ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)
from sklearn.linear_model import Perceptron #as pp

pp = Perceptron()

modelo = pp.fit(X_train,y_train)

y_predict = modelo.predict(X_test)
score = modelo.score(X,Y)

score