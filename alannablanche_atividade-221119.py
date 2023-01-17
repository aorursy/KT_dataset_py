import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn import datasets as dt 

HP = pd.read_csv('/kaggle/input/hepatitis/hepatitis.csv')

HP.head()
HP.describe()
HP.shape
HP.info()
HP.isnull().sum()
idade_media = HP['age'].mean()                      

desvio_padrao = HP['age'].std()                     

str_std = "Desvio Padrão ="+str(round(desvio_padrao,2)) 

str_media = "idade Média ="+str(round(idade_media,2)) 



plt.hist(HP['age'],bins=8, rwidth=0.9)             

plt.title('Histograma da Idade dos Pacientes')

plt.xlabel('idade')

plt.ylabel('contagem')

plt.text(50, 150, str_std)                             

plt.text(50, 200, str_media)

plt.xlim(0, 100)

plt.ylim(0, 500)

plt.show()
HP.keys()
X = HP.loc[:,'age':'histology']

y = HP.loc[:,['class']]
X.head()
y.head()
from sklearn.preprocessing import StandardScaler

SS = StandardScaler()

X = SS.fit_transform(X)
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.linear_model import Perceptron

pc = Perceptron()

modelo = pc.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
score = modelo.score(X_test,y_test)

score