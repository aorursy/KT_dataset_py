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
import sys

import matplotlib.pyplot as plt

import scipy as sp

import IPython

import sklearn

import seaborn as sns

%matplotlib inline
iris_df=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
iris_df.head()
iris_df.info()
iris_df.describe()
sns.pairplot(iris_df)
iris_df.corr()
sns.heatmap(iris_df.corr())
from sklearn.preprocessing import LabelEncoder,StandardScaler
number=LabelEncoder()

iris_df['species']=number.fit_transform(iris_df['species'].astype('str'))

iris_df['species'].dtype
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('species',axis=1),iris_df['species'],

                                                    test_size=0.30,random_state=40)
from sklearn.neighbors import KNeighborsClassifier

error_rate = []



# Will take some time

for i in range(1,20):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))