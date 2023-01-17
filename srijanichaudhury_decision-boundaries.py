# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris
iris = iris.rename(columns = {'SepalLengthCm':'SL','SepalWidthCm':'SW','PetalLengthCm':'PL','PetalWidthCm':'PW'})
iris['Species'].replace('Iris-setosa','0',inplace=True)
iris['Species'].replace('Iris-versicolor','1',inplace=True)
iris['Species'].replace('Iris-virginica','2',inplace=True)
iris_new= iris[['SL','PL','Species']]
X = iris_new.iloc[:,:2].values
Y = iris_new.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
data = DecisionTreeClassifier()
data.fit(x_train,y_train)
xc = np.arange(start = x_train[:,0].min()-1, stop = x_train[:,0].max()+1, step = 0.01)
yc = np.arange(start = x_train[:,1].min()-1, stop = x_train[:,1].max()+1, step = 0.01)

XX,YY = np.meshgrid(xc,yc)
new = np.array([XX.ravel(),YY.ravel()]).T
new.shape
prediction = data.predict(new)

Z1 = prediction.reshape(XX.shape)
plt.contourf(XX,YY,Z1,alpha=0.75)
plt.scatter(x_train[:,0],x_train[:,1])
plt.show()
