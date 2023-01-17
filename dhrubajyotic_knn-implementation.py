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
df=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.shape
df.info()
df.dtypes
df.drop(columns ='Unnamed: 32',inplace=True)
value=[]

for i in df['diagnosis']:

    if i == 'M':

        value.append(1)

    else:

        value.append(0)

df['diagnosis']=value

        
df.head()
X = np.array(df.iloc[:,1:30].values)

y = np.array(df.iloc[:,1].values)
from sklearn.preprocessing import StandardScaler

scaler =  StandardScaler()
X.shape
y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=40,stratify=y)
X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)
np.sqrt(X_train.shape)[0]
k=19

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=k, weights='distance')
from sklearn.metrics import accuracy_score,recall_score
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
recall_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
accuracy=[]

for i in range(1,30):

    knn=KNeighborsClassifier(n_neighbors=i,weights='distance')

    knn.fit(X_train,y_train)

    accuracy.append(recall_score(y_test,knn.predict(X_test)))
plt.plot(range(1,30),accuracy)
k=11

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=k,weights='distance')
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
recall_score(y_test,y_pred) ## overfitting
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
k=3

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=k,weights='distance')

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

recall_score(y_test,y_pred)
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors = 3,weights='distance')

scores = cross_val_score(knn, X, y, cv=5,scoring='recall')# print all 5 times scores 

print(scores)

print(scores.mean())