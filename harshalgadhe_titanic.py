import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/titanic/train_and_test2.csv')

df.head()
df.info()
df.drop(['zero','zero.1','zero.2','zero.3','zero.4','zero.5','zero.6','zero.7','zero.8','zero.9','zero.10','zero.11','zero.12','zero.13','zero.14','zero.15','zero.16','zero.17','zero.18'],axis=1,inplace=True)

df.rename(columns={'2urvived':'Survived'},inplace=True) 

df.head()
df.dropna(inplace=True)
plt.figure(figsize=(12,10))

# we keep annot=True to make the values appear of df.corr() appear on the heatmap

sns.heatmap(df.corr(),annot=True,cmap=plt.cm.plasma)
sns.pairplot(df)
df.columns
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X=df.drop(['Survived'],axis=1)

Y=df['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=300)

lr.fit(X_train,y_train)

yhat=lr.predict(X_test)

print("Accuracy of Logistic Model is:",accuracy_score(yhat,y_test))
from sklearn.metrics import accuracy_score,confusion_matrix

ax=confusion_matrix(yhat,y_test)

sns.heatmap(ax,annot=True,cmap=plt.cm.plasma)

plt.xlabel('Predict')

plt.ylabel('Actual')
from sklearn.neighbors import KNeighborsClassifier

KN=KNeighborsClassifier(n_neighbors=3)

KN.fit(X_train,y_train)

yhat=KN.predict(X_test)

print("Accuracy of K-Nearest Neighbor Model is:",accuracy_score(yhat,y_test))