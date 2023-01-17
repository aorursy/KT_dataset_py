# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/iris/Iris.csv')
data.head()
data=data.iloc[:,[1,3,5]]
data.sample(10)
data['Species'].replace('Iris-setosa',0,inplace=True)
data['Species'].replace('Iris-versicolor',1,inplace=True)
data['Species'].replace('Iris-virginica',2,inplace=True)
data.sample(10)
data['Species'].unique()
X=data.iloc[:,[0,1]]
y=data.iloc[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
sepal_lengths = np.arange(start = X_train[:,0].min()-1,stop=X_train[:,0].max()+1,step=0.01)
petal_lengths = np.arange(start=X_train[:,1].min()-1,stop=X_train[:,1].max()+1,step=0.01)
print(sepal_lengths.shape, petal_lengths.shape)
XX,YY=np.meshgrid(sepal_lenghts,petal_lengths)
print(XX.shape,YY.shape)
inputs = np.array([XX.ravel(), YY.ravel()]).T
labels = clf.predict(inputs)
inputs.shape
labels.shape
plt.contourf(XX,YY,labels.reshape(XX.shape))
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Decision boundary on sepal and petal lengths")
plt.show()
plt.contourf(XX,YY,labels.reshape(XX.shape), alpha=0.5)
plt.scatter(X_train[:,0],X_train[:,1], c=y_train)
