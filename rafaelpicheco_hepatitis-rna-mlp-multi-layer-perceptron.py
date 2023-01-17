import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
hepatitis = pd.read_csv("/kaggle/input/hepatitis/hepatitis.csv")

import matplotlib.pyplot as plt 
hepatitis.describe()
hepatitis.shape
hepatitis.info()
hepatitis.isnull().sum()
idade_media = hepatitis['age'].mean()                     

desvio_padrao = hepatitis['age'].std()                     

str_std = "Desvio Padão ="+str(round(desvio_padrao,2)) 

str_media = "Idade Média ="+str(round(idade_media,2))  

plt.hist(hepatitis['age'],bins=8, rwidth=0.9)              

plt.title('Histograma da Idade dos Pacientes')

plt.xlabel('Idade')

plt.ylabel('contagem')

plt.text(50, 150, str_std)                            

plt.text(50, 200, str_media)

plt.xlim(0, 100)

plt.ylim(0, 500)

plt.show()
hepatitis.columns
X = hepatitis.loc[:,:]

y = hepatitis.loc[:,'class']

x.shape
hepatitis.keys()
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

x = ss.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
from sklearn.linear_model import Perceptron

pp = Perceptron()

modelo = pp.fit(x_train,y_train)

y_predict = modelo.predict(x_test)
score = modelo.score(X,y)

score