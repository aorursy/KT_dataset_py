# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import os
#print(os.listdir("../input"))
mushroom_class = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
mushroom_class.head()
mushroom_class.tail()
pd.crosstab(index=mushroom_class['class'],columns=['count'],dropna='False')
mushroom_class.isnull().sum()
mushroom_class['class'].unique()
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in mushroom_class.columns:
    mushroom_class[col] = labelencoder.fit_transform(mushroom_class[col])
 
mushroom_class.head()
mushroom_class.info()
y = mushroom_class['class']
X = mushroom_class.drop(['class'], axis = 1)
X.head()
#from sklearn.preprocessing import OneHotEncoder

#onehotencoder = OneHotEncoder(categorical_features = 'all',sparse=False) 
#X = onehotencoder.fit_transform(X).toarray() 


X = pd.get_dummies(X,columns = X.columns)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#X_train.head()
y_train.head()
clf = GradientBoostingClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred
clf.score(X_test,y_test)
from sklearn.metrics import accuracy_score
print("The accuarcy score of decision tree classifier is ",accuracy_score(y_test,y_pred))
cv_score_ten = cross_val_score(clf,X,y,cv=10)
cv_score_ten.mean()