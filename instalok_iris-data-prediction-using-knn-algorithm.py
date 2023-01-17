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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from pandas import Series,DataFrame

import scipy

from pylab import rcParams

import urllib

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn import neighbors

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings('ignore')

print ('Setup Complete')
df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
df.shape
df.head(5)
df.dtypes
df.isnull().sum()
df.info()
df.describe()
plt.figure(figsize=(12,12))

sns.heatmap(df.drop('species',axis=1).corr(),annot=True)
sns.set(style="ticks")

sns.pairplot(df, hue="species")
X_prime=df.ix[:,(0,1,2,3)].values

y=df.ix[:,4].values

X= preprocessing.scale(X_prime)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3, random_state  = 5)
#K-Nearest Neighbours

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy is:',accuracy_score(y_pred,y_test))
pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])
scores=[]

for n in range(1,15):

    model=KNeighborsClassifier(n_neighbors=n)

    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    scores.append(accuracy_score(y_pred,y_test))

    

plt.plot(range(1,15),scores)

plt.xlabel("Number of neighbors")

plt.ylabel("Accuracy")

plt.show()
model = KNeighborsClassifier(n_neighbors=6)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy is:',accuracy_score(y_pred,y_test))
pd.crosstab(y_test, y_pred, rownames=['variety'], colnames=['predicted'])