import os

for dirname, _, filenames in os.walk('/kaggle/input/creditscreening/credit.data'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import numpy as np 

import pandas as pd

import os

import sklearn.datasets as dt

import matplotlib.pyplot as plt

import seaborn as sn

sn.set()
import pandas as pd

nomes = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

credit = pd.read_csv("/kaggle/input/creditscreening/credit.data", sep = ",", names = nomes, na_values = '?')

credit.head()
credit.describe()
credit.isnull().sum()
credit.dtypes
grupo = ['A1','A4','A5','A6','A7']

for i in grupo: 

    credit[i]=credit[i].fillna(value=credit[i].mode)
credit.isnull().sum()
grupo = ['A2','A14']

for i in grupo: 

    credit[i]=credit[i].fillna(value=credit[i].mean)
credit.isnull().sum()
grupo = ['A1','A4','A5','A6','A7','A9','A10','A12','A13']

for i in grupo:

    credit[i] = credit[i].astype(str)

    credit[i] = credit[i].astype('category')
credit.dtypes
credit['A16'] = credit['A16'].astype('category')

credit['A16'].cat.codes
credit.dtypes
credit = pd.get_dummies(credit,columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
dic = dt.load_digits()

dic.keys()
X = dic.data

Y = dic.target



from sklearn.model_selection import train_test_split 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

X_train.shape
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

modelo = knn.fit(X_train,Y_train)

y_pred = modelo.predict(X_test)

y_score = modelo.score(X_test,Y_test)

y_score
compara = pd.DataFrame(Y_test)

compara['predictKNN'] = y_pred

compara.head(20)