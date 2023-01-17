# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

%matplotlib inline

## Import dataset 

df = pd.read_csv('/kaggle/input/heart-disease-dataset/heart.csv')
df.head()
df.info()
df.describe()
sns.pairplot(df,hue='target')
sns.catplot(x='sex',y='age',data=df,hue='target',palette='pastel')

plt.xlabel('0=Female, 1=Male')
sns.catplot(x='cp',y='age',data=df,hue='target')



sns.relplot(x='age',y='thalach',hue='target',data=df,kind='line')
sns.relplot(x='age',y='thalach',hue='sex',data=df,kind='line')



sns.catplot(x='sex',y='age',data=df,hue='target',kind='box')
sns.catplot(x='trestbps',y='target',row='sex',kind='box',orient='h',height=1.5,aspect=5,data=df,palette='pastel')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X = df.drop('target',axis=1)

y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                test_size=0.30, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
from sklearn.metrics import classification_report, confusion_matrix



predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.neighbors import KNeighborsClassifier
X = df.drop('target',axis=1)

y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                test_size=0.30, random_state=101)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
error_rate = []

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