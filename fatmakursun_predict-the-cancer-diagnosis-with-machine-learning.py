# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/breastCancer.csv")
df.head()
df.info()
df.drop(columns=['Unnamed: 32'],inplace=True)
df.drop(columns=['id'],inplace=True)
set(df['diagnosis'])
df['diagnosis'] =[ 1 if i =='M' else 0 for i in df['diagnosis']]
plt.figure(figsize=(12,5))

sns.countplot(x= 'diagnosis', data=df)
df.plot(figsize=(18,8))
plt.figure(figsize=(12,7))

df['smoothness_mean'].hist(bins=30,color='darkred',alpha=0.7)
df['perimeter_worst'].hist(color='blue',bins=40,figsize=(8,4))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('diagnosis',axis=1))
scaled_features = scaler.transform(df.drop('diagnosis',axis=1))
df.columns
df_feat = pd.DataFrame(scaled_features,columns=df.columns[1:])

df_feat.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['diagnosis'],test_size=0.20,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

pred
df['diagnosis'].head(5)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1

knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=1')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
# NOW WITH K=11

knn = KNeighborsClassifier(n_neighbors=11)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=11')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))