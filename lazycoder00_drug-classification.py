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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/kaggle/input/drug-classification/drug200.csv')
data.head(5)
data.Drug.unique()
data.isnull().sum()
data.shape
data.columns
sns.violinplot(x=data.BP, y=data.Age, data=data)
sns.boxplot(x=data.Sex, y=data.Age, data=data)
sns.boxplot(x=data.Cholesterol, y=data.Age, data=data)
sns.countplot(x=data.Sex,data=data)
sns.countplot(x=data.Cholesterol,data=data)
sns.countplot(x=data.Drug,data=data)
lb=LabelEncoder()
data.Sex=lb.fit_transform(data.Sex)
data.BP=lb.fit_transform(data.BP)
data.Cholesterol=lb.fit_transform(data.Cholesterol)
data.head(3)
BP=data['BP']
BP_=pd.get_dummies(BP,drop_first=True)
BP_.head(2)
y=data['Drug']
y=lb.fit_transform(y)
data.drop(['BP','Drug'],axis=1,inplace=True)

new=pd.concat([data,BP_],axis=1)
sns.heatmap(new.corr(),annot=True,fmt='.1f',linewidths=0.5)
X=new
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,shuffle=True)
rf=RandomForestClassifier()
knn=KNeighborsClassifier()
dtc=DecisionTreeClassifier()
rf.fit(X_train,y_train)
knn.fit(X_train,y_train)
dtc.fit(X_train,y_train)
y_1=rf.predict(X_test)
y_2=knn.predict(X_test)
y_3=dtc.predict(X_test)

print('accuracy score 1: ',accuracy_score(y_test,y_1))
print('accuracy score 2: ',accuracy_score(y_test,y_2))
print('accuracy score 3: ',accuracy_score(y_test,y_3))
cmap1=confusion_matrix(y_test, y_1)
cmap2=confusion_matrix(y_test, y_2)
cmap3=confusion_matrix(y_test, y_3)
sns.heatmap(cmap1,annot=True)

sns.heatmap(cmap2,annot=True)

sns.heatmap(cmap3,annot=True)

xGb = xgboost.XGBClassifier()
xGb.fit(X_train,y_train)
y_testpred= xGb.predict(X_test)

cmap1=confusion_matrix(y_test, y_testpred)
sns.heatmap(cmap1,annot=True)
print('Accuracy is: ', accuracy_score(y_test,y_testpred))
