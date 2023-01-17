# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input/"))

import warnings

warnings.filterwarnings('ignore')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris=pd.read_csv('/kaggle/input/iris/Iris.csv')
iris
iris=iris[['SepalLengthCm','PetalLengthCm','Species']]
iris.rename(columns={'SepalLengthCm':'SL','PetalLengthCm':'PL'},inplace=True)
iris
iris['Species'].replace({'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3},inplace=True)
X=iris.iloc[:,:-1].values

y=iris.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()



X = ss.fit_transform(X)
print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
a=np.arange(start=X_train[:,0].min()-1, stop=X_train[:,0].max()+1, step=0.01)

b=np.arange(start=X_train[:,1].min()-1, stop=X_train[:,1].max()+1, step=0.01)

XX,YY=np.meshgrid(a,b)
print(XX.shape)

print(YY.shape)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)
pred=dt.predict(np.array([XX.ravel(),YY.ravel()]).T)
print(pred.shape)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,10))

plt.contourf(XX,YY,pred.reshape(XX.shape));
plt.figure(figsize=(15,10))

plt.contourf(XX,YY,pred.reshape(XX.shape))

sns.scatterplot(X_train[:,0],X_train[:,1], hue=y_train, style=y_train, legend='full', palette=['purple','blue','orange'])

plt.show()