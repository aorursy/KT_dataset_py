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
df=pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')
df.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.countplot(df['target_class'])
df.info()
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()

scaler.fit(df.drop('target_class', axis=1))

scaled_feat=scaler.transform(df.drop('target_class', axis=1))

df1=pd.DataFrame(scaled_feat, columns=df.columns[:-1])

df1.head()
from sklearn.model_selection import train_test_split
X=df1

y=df['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred= knn.predict(X_test)
sns.distplot(pred-y_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))
error_rate=[]



for i in range(1,100):

    knn= KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i=knn.predict(X_test)

    error_rate.append(np.mean(pred_i!=y_test))
figure=plt.figure(figsize=(12,7))

error_df=pd.DataFrame(error_rate, index=range(1,100))

plt.plot(error_df,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)
knn= KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)

pred=knn.predict(X_test)

print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))

    