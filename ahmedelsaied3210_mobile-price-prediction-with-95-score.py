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

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
data=pd.read_csv('../input/mobile-price-classification/train.csv')

test=pd.read_csv('../input/mobile-price-classification/test.csv')
fig = plt.figure(figsize = (15,20))

ax = fig.gca()

data.hist(ax = ax)
print(data.shape)

data.head()
print(test.shape)

test.head()
data.columns
data.isnull().sum()
data.dtypes
X=data.drop(['price_range'],axis=1)
Y=data['price_range']
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=4,test_size=0.2)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)
from sklearn.metrics import mean_squared_error

predict=lr.predict(x_test)

mean_squared_error(predict,y_test)
lr.score(x_test,y_test)
plt.scatter(y_test,predict)

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(x_train,y_train)
knn_predict=knn.predict(x_test)

print(knn.score(x_test,y_test))

print(mean_squared_error(knn_predict,y_test))
test=test.drop(['id'],axis=1)

test_predict=knn.predict(test)
print(test_predict)
pd.DataFrame(test_predict).head()
test['price_range']=test_predict

test.head(50)