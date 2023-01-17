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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
data=pd.read_csv('../input/glass/glass.csv')

data.head()
data.isnull().sum()
data.info
data.columns
data.shape
data.describe()
fig = plt.figure(figsize = (15,20))

ax = fig.gca()

data.hist(ax = ax)
X=data.drop(['Type'],axis=1)

X.head()
Y=data['Type']
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=4,test_size=0.20)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)
from sklearn.metrics import mean_squared_error

predict=lr.predict(x_test)

print(mean_squared_error(predict,y_test))

print(lr.score(x_test,y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(x_train,y_train)
knn_predict=knn.predict(x_test)

print(knn.score(x_test,y_test))

print(mean_squared_error(knn_predict,y_test))
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

knn_predict=dt.predict(x_test)

print(dt.score(x_test,y_test))

print(mean_squared_error(knn_predict,y_test))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis()

lda.fit(x_train,y_train)

knn_predict=lda.predict(x_test)

print(lda.score(x_test,y_test))

print(mean_squared_error(knn_predict,y_test))