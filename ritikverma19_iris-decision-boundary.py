# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/iris/Iris.csv")
data.head()
data = data.iloc[:, [2, 3, 5]]
data.head()
data['Species'].replace('Iris-setosa','0',inplace=True)
data['Species'].replace('Iris-versicolor','1',inplace=True)
data['Species'].replace('Iris-virginica','2',inplace=True)
data.shape
data.head()
X=data.iloc[:,0:2].values
y=data.iloc[:,2].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
print(X_train.shape)
print(X_test.shape)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
a = np.arange(start = X_train[:,0].min()-1,stop=X_train[:,0].max()+1,step=0.01)
b = np.arange(start=X_train[:,1].min()-1,stop=X_train[:,1].max()+1,step=0.01)
a.shape
b.shape
XX, YY = np.meshgrid(a, b)
XX.shape
YY.shape
x = np.array([XX.ravel(), YY.ravel()]).T
labels = clf.predict(x)
labels.shape
plt.contourf(XX,YY,labels.reshape(XX.shape))
plt.xlabel("Sepal Length in Cm")
plt.ylabel("Petal Length in Cm")
plt.title("VISUALIZING DECISION BOUNDARY FOR IRIS DATASET")
plt.show()
X_train[:,0].shape
X_train[:,1]
# plt.contourf(XX,YY,labels.reshape(XX.reshape),alpha=0.75)
plt.contourf(XX,YY,labels.reshape(XX.shape), alpha=0.75)
plt.scatter(X_train[:,0],X_train[:,1])