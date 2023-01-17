# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_csv('../input/Mall_Customers.csv')
df.info()
x=df.iloc[:,[3,4]].values

y=df.iloc[:,1].values
from sklearn.cluster import KMeans

value=[]

for i in range(1,11):

    cluster=KMeans(n_clusters=i)

    cluster.fit(x)

    value.append(cluster.inertia_)

print(value)

plt.plot(value)

plt.title('Elbow Method')

plt.xlabel('No.of clusters')

plt.ylabel('wcss value')
cluster=KMeans(n_clusters=5)

group=cluster.fit_predict(x)
plt.figure(figsize=(10,6))

plt.scatter(x[group==0,0],x[group==0,1],color='red',label='people who spend wisely')

plt.scatter(x[group==1,0],x[group==1,1],color='black',label='rich people')

plt.scatter(x[group==2,0],x[group==2,1],color='green',label='people who love to shop')

plt.scatter(x[group==3,0],x[group==3,1],color='blue',label='people with lot of savings')

plt.scatter(x[group==4,0],x[group==4,1],color='yellow',label='people who dont shop alot')

plt.title('Groups of different people')

plt.xlabel('Income')

plt.ylabel('Score')

plt.legend()

plt.show()
from sklearn.preprocessing import LabelEncoder

label_x=LabelEncoder()

y=label_x.fit_transform(y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

classifier=DecisionTreeClassifier(criterion='entropy')

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)

score=accuracy_score(y_test,y_pred)

print(cm)

print("Decision Tree Classifier accuracy score is ",score)
from sklearn.ensemble import RandomForestClassifier

classifier= RandomForestClassifier(n_estimators=100)

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)

print(cm)

score=accuracy_score(y_test,y_pred)

print("Random Forest Classifier accuracy score is ",score)