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



print(os.listdir("../input"))

df = pd.read_csv("../input/bank-full.csv") 

df.head()



# Get total number of rows in the dataset/dataframe

print(len(df))

# Any results you write to the current directory are saved as output.



df.info()

# Any results you write to the current directory are saved as output.
cols = df.columns[df.isnull().any()]
#label encoding also try

#pruning

df.drop(['day', 'month'], axis=1, inplace=True)

df= pd.get_dummies(df, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'])

df.head()
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier 

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

import math

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

parameters = {'max_depth':[2,5,7,11], 'min_samples_split':[2,4,6], 'min_samples_leaf':[1,2,3], 'max_leaf_nodes': [2,4,6,8]}





x = df.loc[:,df.columns!='y']

# print(x.shape)

y = df['y'].values

# print(y.shape)

# x= preprocessing.normalize(x)

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.3, random_state = 7)



# clf = DecisionTreeClassifier()



# dtree_gscv = GridSearchCV(clf, parameters, cv=3)

# dtree_model = dtree_gscv.fit(x_train,y_train)

# print(dtree_model.best_params_)

clf = DecisionTreeClassifier(max_depth= 5, max_leaf_nodes= 8, min_samples_leaf= 1, min_samples_split=2)

clf=clf.fit(x_train,y_train)



print(clf.score(x_train, y_train))

print(clf.score(x_test, y_test))



model = BaggingClassifier(n_estimators=100)

model.fit(x_train,y_train)

print(model.score(x_train, y_train))

print(model.score(x_test, y_test))



rf = RandomForestClassifier(n_estimators=100)

rf = rf.fit(x_train,y_train)

print(rf.score(x_train, y_train))

print(rf.score(x_test, y_test))



# print(clf.predict(x_test))

# print((y_test))

#rint("Accuracy:",metrics.accuracy_score(x_train, y_train))

#print("Accuracy:",metrics.accuracy_score(x_test, y_test))