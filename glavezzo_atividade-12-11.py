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

        

import matplotlib.pyplot as plt

import seaborn as sn

sn.set()
import pandas as pd

nomes = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

df = pd.read_csv("/kaggle/input/creditscreening/credit-screening.data",sep=",", names=nomes, na_values= '?')

df.head()
df.describe()
df.isnull().sum()
cols = ['A1', 'A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13','A1', 'A4']

for i in cols:

    df[i]=df[i].fillna(value=df[i].mode)



#df = df.fillna(df.mode(), inplace=True)
cols = ['A2', 'A14']

for i in cols:

    df[i]=df[i].fillna(value=df[i].mean)

cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']

for i in cols:

    df[i]= df[i].astype(str)

    df[i]= df[i].astype('category')
df['A16'] = df['A16'].astype('category')

df['A16'].cat.codes
df = pd.get_dummies( df, columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xss = ss.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(Xss,y, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

modelo= knn.fit(X_train,y_train)

y_pred = modelo.predict(X_test)

y_score = modelo.score(X_test,y_test)

y_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dtc = DecisionTreeClassifier()

modelo = dtc.fit(X_train,y_train)

y_pred = modelo.predict(X_test)

import graphviz as gpz

dot_data = export_graphviz(dtc)

graph = gpz.Source(dot_data)

graph