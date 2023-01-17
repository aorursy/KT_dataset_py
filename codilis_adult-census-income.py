# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/adult.csv')

df.head()

for col in df:

    x = df[col].value_counts()

    if '?' in x:

        print(col, x['?'])
len(df)

df.drop(df[df['native.country']=='?'].index, inplace=True)

for col in df:

    x = df[col].value_counts()

    if '?' in x:

        print(col, x['?'])
print(df['relationship'].value_counts())

import matplotlib.pyplot as plt

x = {'Husband':[0, 0], 'Not-in-family':[0, 0], 'Own-child':[0, 0], 

     'Unmarried':[0, 0], 'Wife':[0, 0], 'Other-relative':[0, 0]}

y = {'<=50K':0, '>50K':1}

for i in range(len(df)):

    x[df['relationship'][i]][y[df['income'][i]]] += 1

print(x)

a = list(x.keys())

b = list(x.values())

m = []

n = []

for i in b:

    m.append(i[0])

    n.append(i[1])

p1 = plt.bar(a, m)

p2 = plt.bar(a, n)

plt.legend((p1[0], p2[0]), ('<=50K', '>50K'))

plt.show()
df['workclass'].replace('?', df['workclass'].value_counts().index[0], inplace=True)

df['occupation'].replace('?', df['occupation'].value_counts().index[0], inplace=True)
df.drop(['fnlwgt', 'education', 'relationship', 'race'], axis=1, inplace=True)

df.head()
X = df.iloc[:, 0:-1].values

y = df.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

for i in [1, 3, 4, 5, 9]:

    X[:, i] = labelencoder_X.fit_transform(X[:, i])

onehotencoder = OneHotEncoder(categorical_features = [1, 3, 4, 5, 9])

X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable

# labelencoder_y = LabelEncoder()

# y = labelencoder_y.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

dtc = DecisionTreeClassifier( min_samples_split=200)

dtc.fit(X_train, y_train)

pre = dtc.predict(X_test)

print(accuracy_score(pre, y_test))
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score

etc = ExtraTreesClassifier(n_estimators=9, random_state=111)

etc.fit(X_train, y_train)

pre = etc.predict(X_test)

print(accuracy_score(pre, y_test))
