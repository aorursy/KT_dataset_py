import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

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
from sklearn import datasets as dt 

dados = pd.read_csv('/kaggle/input/hepatitis/hepatitis.csv')

dados.head()
dados.describe()
dados.shape
dados.info()
dados.isnull().sum()
import matplotlib.pyplot as plt

plt.hist(dados['age'], bins=8, rwidth=0.9)

plt.title("Histograma Idade Doentes")

plt.xlabel("Idade")

plt.ylabel("contagem")

plt.xlim(0,80)

plt.ylim(0,80)

plt.show()
dados.keys()
X = dados.loc[:,'age':'histology']

y = dados.loc[:,['class']]
X.head()
y.head()
#Utilize a biblioteca de pré-processamento para normalizar os atributos de entrada (X)

from sklearn.preprocessing import StandardScaler

SS = StandardScaler()

X = SS.fit_transform(X)
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(X_train, y_train)

y_predict = modelo.predict(X_test)
score = modelo.score(X_test, y_test)

score