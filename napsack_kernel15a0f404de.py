# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
y  = train.iloc[:,1].values

X = train.iloc[:,[2,4,5,6,7,9,11]].values

X_test = test.iloc[:,[1,3,4,5,6,8,10]].values
#visualization 

X[1,:]

train.head(5)

plt.scatter(X[y == 0,0],X[y == 0,3], c = 'r',label = "Death")

plt.scatter(X[y == 1,0],X[y == 1,3], c = 'g',label = "life")

plt.show()
from sklearn.preprocessing import Imputer

imp = Imputer()

temp = X[:,2]

temp = temp.reshape(-1,1)

temp = imp.fit_transform(temp)

X[:,2] = temp.ravel()

del(temp)

X_test[:,[2,5]] = imp.fit_transform(X_test[:,[2,5]])

tesst = pd.DataFrame(X[:,[1,6]]) 

tesst.describe()

tesst[1] = tesst[1].fillna('S')



X[:,[1,6]] = tesst

del(tesst)

tesst = pd.DataFrame(X_test[:,[1,6]]) 

tesst.describe()

tesst[1] = tesst[1].fillna('S')

tesst[0] = tesst[0].fillna('male')





X_test[:,[1,6]] = tesst
from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()

X[:,1] = lab.fit_transform(X[:,1])

X[:,6] = lab.fit_transform(X[:,6])

X_test[:,1] = lab.fit_transform(X_test[:,1])

X_test[:,6] = lab.fit_transform(X_test[:,6])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

X_test = sc.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X,y)

log_reg.score(X,y)



Y_pred = log_reg.predict(X_test)
print(log_reg.score(X,y))
