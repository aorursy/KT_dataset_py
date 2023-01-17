# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris=pd.read_csv('/kaggle/input/iris/Iris.csv')
iris
iris.rename(columns={'SepalLengthCm':'SL','PetalLengthCm':'PL'},inplace=True)
X=iris[['SL','PL']]
X
y=iris['Species']
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train
y_train
X_test
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred,y_test)
x_axis=np.linspace(X_train['SL'].min()-1,X_train['SL'].max()+1,900)
y_axis=np.linspace(X_train['PL'].min()-1,X_train['PL'].max()+1,1000)
XX,YY=np.meshgrid(x_axis,y_axis)
YY.shape
points=np.array([XX.ravel(),YY.ravel()]).T
pred=clf.predict(points)
Z=pred.reshape(XX.shape)
Z
Z1
plt.subplots(figsize=(20,10))
plt.contourf(XX,YY,Z1,alpha=0.75)
plt.scatter(X_train['SL'],X_train['PL'],c=np.where(y_train=='Iris-setosa',0,np.where(y_train=='Iris-versicolor',1,2)))
plt.title('Decision Boundary for Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()
plt.show()
