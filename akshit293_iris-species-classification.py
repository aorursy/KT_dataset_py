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
data=pd.read_csv('../input/iris/Iris.csv')
data.head()
data.info()
data.Species.value_counts()
data['Species']=data['Species'].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])
data.Species.value_counts()
y=pd.DataFrame()
y['Species']=data['Species']
data.drop('Species',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train,X_test,y_train,y_test=train_test_split(data,y,random_state=0)
clf=RandomForestClassifier(max_depth=3,random_state=0)
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
from sklearn.tree import DecisionTreeClassifier
clf2=DecisionTreeClassifier(max_depth=3,random_state=0)
clf2.fit(X_train,y_train)
print(clf2.score(X_train,y_train))
print(clf2.score(X_test,y_test))
from sklearn.linear_model import LogisticRegression
clf3=LogisticRegression()
clf3.fit(X_train,y_train)
print(clf3.score(X_train,y_train))
print(clf3.score(X_test,y_test))
print(clf3.coef_)
print(clf3.intercept_)
import seaborn as sns
scores=pd.DataFrame({'Model':['Random Forest','Decision Tree','Logistic Regression'],'Score':[clf.score(X_test,y_test),clf2.score(X_test,y_test),clf3.score(X_test,y_test)]})
sns.catplot(x='Model',y='Score',data=scores,kind='bar')
