# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mat

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris=pd.read_csv('/kaggle/input/iris/Iris.csv')

iris.sample(10)

iris['Id']
data=iris.drop(['Id','SepalWidthCm','PetalWidthCm'],axis=1)
data.sample(10)

data['Species'].replace({'Iris-setosa':'0','Iris-versicolor':'1','Iris-virginica':'2'},inplace=True)
X = data.iloc[:,:2].values

Y = data.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=9)
x_test.shape
x_train.shape
scaler = StandardScaler()

x_train=scaler.fit_transform(x_train)

x_test=scaler.fit_transform(x_test)
clf = DecisionTreeClassifier(criterion='entropy',splitter='best')

clf.fit(x_train,y_train)
knn = KNeighborsClassifier(n_neighbors = 13, algorithm = 'auto')

knn.fit(x_train,y_train)
x1 = np.arange(start = x_train[:,0].min()-1, stop = x_train[:,0].max()+1, step = 0.01)

y1 = np.arange(start = x_train[:,1].min()-1, stop = x_train[:,1].max()+1, step = 0.01)

print(x1.shape[0])

print(y1.shape[0])

print("total = ",x1.shape[0]*y1.shape[0])
xx,yy = np.meshgrid(x1,y1)
xx.shape
yy.shape
data1 = np.array([xx.ravel(),yy.ravel()]).T
data1.shape
y_pred = knn.predict(data1)
y_pred
plt.figure(figsize=(25,10))

plt.contourf(xx,yy,y_pred.reshape(xx.shape))

plt.xlabel("Sepal Length")

plt.ylabel("Petal Length")

plt.title("DECISION BOUNDARY")

plt.show()
plt.figure(figsize=(25,15))

plt.contourf(xx,yy,y_pred.reshape(xx.shape),alpha=0.75)

plt.scatter(x_train[:,0],x_train[:,1])

plt.xlabel("Sepal Length")

plt.ylabel("Petal Length")

plt.title("DECISION BOUNDARY")

plt.show()