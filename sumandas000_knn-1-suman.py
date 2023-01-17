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
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head(5)
data.drop(['Unnamed: 32'],axis=1,inplace=True) #useless column
data['diagnosis']=data['diagnosis'].str.replace('B','0')
data['diagnosis']=data['diagnosis'].str.replace('M','1')
data['diagnosis']=data['diagnosis'].astype(int)
data.shape
data.sample(10)
X=data.iloc[:,2:33].values
y=data.iloc[:,1].values
#split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#method 1
k=np.sqrt(X_train.shape[0])
k
k=21
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
#predict
y_pred=knn.predict(X_test)
#accuracy check
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)
#mathod 2

accuracy=[]
for i in range(1,51):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    accuracy.append(accuracy_score(y_pred,y_test))
accuracy
import matplotlib.pyplot as plt
plt.plot(range(1,51),accuracy)
#taken k=10
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_pred,y_test)