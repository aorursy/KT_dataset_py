import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
atv = pd.read_csv('/kaggle/input/hepatitis/hepatitis.csv') 

atv.head()  
atv.describe()
atv.shape
atv.info
atv.isnull().sum()
import matplotlib.pyplot as plt 

plt.hist(atv['age'],bins=8, rwidth=0.9)              

plt.title('Histograma da Idade dos Passageiros')

plt.xlabel('Idade')

plt.ylabel('contagem')

plt.show()
atv.keys()

x = atv.loc[:,'age':'histology']

y = atv.loc[:, ['class']]
from sklearn.preprocessing import StandardScaler

ss = StandardScaler() 

x = ss.fit_transform(x)  
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(x_train, y_train)

y_predict = modelo.predict(x_test)
score = modelo.score(x,y)

score