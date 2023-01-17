import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/hepatitis/hepatitis.csv')

df.head
df.describe()
df.shape
df.info()
df.isnull(),df.sum()
import matplotlib.pyplot as plt

plt.hist(df['age'], bins=None, rwidth=0.9 )

plt.title('Distribuição de Idades')

plt.xlabel('idade')

plt.ylabel('contagem')

plt.show
X = df.loc[:,'age':'histology']

y = df['class']
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
score = modelo.score(X,y)

score
import pandas as pd

hepatitis = pd.read_csv("../input/hepatitis/hepatitis.csv")