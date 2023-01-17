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
df=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.isnull().any()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df['quality'].value_counts().plot(kind='bar')
cor=df.corr()
cor
sns.heatmap(cor)
x=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]
y=df.iloc[:,[11]]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=5,test_size=0.2)
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
x_train.shape
x_test.shape

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
pred
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
import xgboost
xgb=xgboost.XGBClassifier()
xgb.fit(x_train,y_train)
pre=xgb.predict(x_test)
print(accuracy_score(y_test,pre))
print(confusion_matrix(y_test,pre))
from sklearn.tree import DecisionTreeClassifier
tr= DecisionTreeClassifier()
tr.fit(x_train,y_train)
pr=tr.predict(x_test)
print(accuracy_score(y_test,pr))
print(confusion_matrix(y_test,pr))

