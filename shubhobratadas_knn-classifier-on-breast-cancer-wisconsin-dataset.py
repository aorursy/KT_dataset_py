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
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')



data.head()
data.info()
data.shape
data['diagnosis'].value_counts()
X=data.iloc[:,2:32].values
X
y=data.iloc[:,1].values



y
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)



X_train
X_test=scaler.fit_transform(X_test)



X_test
np.sqrt(X_train.shape[0])
k=21



from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_jobs=k)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)



y_pred.shape
y_test.shape
from sklearn.metrics import accuracy_score



accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_test,y_pred)
accuracy=[]



for i in range(1,50):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    accuracy.append(accuracy_score(y_test,knn.predict(X_test)))

    
accuracy
len(accuracy)
plt.plot(range(1,50),accuracy)
k=3



from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_pred,y_test)