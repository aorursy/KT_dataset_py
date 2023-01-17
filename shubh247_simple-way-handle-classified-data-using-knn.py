# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read the file

df = pd.read_csv("../input/Classified Data")
df.head()
df = df.drop("Unnamed: 0", axis=  1)

df.info()
df.describe()
from sklearn.model_selection import train_test_split
#drop trget class from frature

X = df.drop('TARGET CLASS',axis=1)

y = df['TARGET CLASS']
#train the model

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#import knn library

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

#fit the model

knn.fit(X_train, y_train)

#predict the model

y_pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#fit standard scaler

scaler.fit(df.drop('TARGET CLASS',axis=1))  #drop target class
#removing noise

scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

scaled_features
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
X = df_feat  #feature

y = df['TARGET CLASS']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
error_rate = []



for i in range(1,20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    y_pred_i = knn.predict(X_test)

    error_rate.append(np.mean(y_pred_i != y_test))
plt.figure(figsize=(10,5))

plt.plot(range(1,20),error_rate,color='blue',ls='--',marker='o',markerfacecolor='red', markersize=10)

plt.title('Error Rate vs K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors = 11)

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))