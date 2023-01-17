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
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
df
df=df.drop(columns=['id','Unnamed: 32'])
df
df.isna().sum()
X=df.drop(columns=['diagnosis'])
y=df['diagnosis']
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
new=s.fit_transform(X)
new
new.shape
transformed_X=pd.DataFrame(new)
transformed_X
transformed_X.columns=X.columns
transformed_X
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(transformed_X,y,test_size=0.2)
import sklearn
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
linear=LinearRegression()
linear.fit(X_train,y_train)
linear.score(X_test,y_test)
lg=LogisticRegression()
lg.fit(X_train,y_train)
lg.score(X_test,y_test)
svc=SVC()
svc.fit(X_train,y_train)
svc.score(X_test,y_test)
tree=DecisionTreeClassifier(criterion='entropy')
tree.fit(X_train,y_train)
tree.score(X_test,y_test)
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf.score(X_test,y_test)
