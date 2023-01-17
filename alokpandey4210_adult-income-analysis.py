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
columns=['Age','Workloss','fnlgwt','Education','EducationNum','MaritalStatus','Occupation','Relationship','Race','Sex','CapitalGain','CapitalLoss','Hours/week','country','Above/Below50K']

df=pd.read_csv('/kaggle/input/us-census-data/adult-training.csv',names=columns)

df
df.corr(method='pearson')
X=df.drop(['Above/Below50K'],axis=1)
y=df['Above/Below50K']
X.isnull().sum()
X=pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
from sklearn.linear_model import LogisticRegression

d=LogisticRegression()

d.fit(X_train,y_train)

d.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

s=KNeighborsClassifier()

s.fit(X_train,y_train)

s.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

m=DecisionTreeClassifier()

m.fit(X_train,y_train)

m.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

l=RandomForestClassifier()

l.fit(X_train,y_train)

l.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesClassifier

q=ExtraTreesClassifier()

q.fit(X_train,y_train)

q.score(X_test,y_test)