# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
pd.set_option('display.MAX_COLUMNS',None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data
data.shape
data.drop(columns='Unnamed: 32',inplace=True)
X=data.iloc[:,2:].values
X.shape
y=data['diagnosis']
y.shape
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.neighbors import  KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=24)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)
accuracy=[]
for i in range(1,31):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    accuracy.append(accuracy_score(y_pred,y_test))
plt.plot(range(1,31),accuracy)
np.argmax(accuracy)+1
from sklearn.metrics import confusion_matrix
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(accuracy_score(y_pred,y_test))
confusion_matrix(y_pred,y_test,labels=['M','B'])
accuracy=[]
for j in range(50):
    knn=KNeighborsClassifier(n_neighbors=9)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    accuracy.append(accuracy_score(y_pred,y_test))
np.mean(accuracy)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
accuracy_score(y_test,y_pred)
dtc.get_depth()
accuracy=[]
for j in range(1,8):
    dtc=DecisionTreeClassifier(max_depth=j)
    dtc.fit(X_train,y_train)
    y_pred=dtc.predict(X_test)
    accuracy.append(accuracy_score(y_pred,y_test))
plt.plot(range(1,8),accuracy)
np.argmax(accuracy)+1
max(accuracy)
