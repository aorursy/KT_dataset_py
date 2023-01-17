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
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nomes = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

df = pd.read_csv("/kaggle/input/creditscreening/credit.data", sep=",", names=nomes, na_values = '?')

df.head()
df.describe()
df.isnull().sum()
df.mode()
df['A1'] = df['A1'].fillna(df['A1'].mode)

df['A4'] = df['A4'].fillna(df['A4'].mode)

df['A5'] = df['A5'].fillna(df['A5'].mode)

df['A6'] = df['A6'].fillna(df['A6'].mode)

df['A7'] = df['A7'].fillna(df['A7'].mode)
df.isnull().sum()
df['A2'] = df['A2'].fillna(df['A2'].mean)

df['A14'] = df['A14'].fillna(df['A14'].mean)
df.isnull().sum()
cols = ['A1','A4','A5','A6','A7','A9','A10','A12','A13','A16']

for i in cols:

    df[i]=df[i].astype(str)

    df[i]=df[i].astype('category')



df.info()

        
df['A16']=df["A16"].cat.codes
df['A16']
X = df.loc[:,:'A15']

y = df.loc[:,['A16']]
X
X = pd.get_dummies(x,columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
X.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

modelo = knn.fit(X_train, y_train)

y_pred = modelo.predict(X_train)

y_score = modelo.score(X_train, y_train)

y_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dtc = DecisionTreeClassifier()

modelo = dtc.fit(X_train,y_train)



y_pred = modelo.predict(X_test)