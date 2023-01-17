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
df = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-por.csv')

df.info()
df.head(3)
df.isnull().sum()
df.info()
object_type_features = df.select_dtypes("object").columns

object_type_features
df.G3.value_counts()
'''

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



df.school = le.fit_transform(df.school)

df.sex = le.fit_transform(df.sex)

df.address = le.fit_transform(df.address)

df.famsize = le.fit_transform(df.famsize)

df.Pstatus = le.fit_transform(df.Pstatus)

df.Mjob = le.fit_transform(df.Mjob)

df.Fjob = le.fit_transform(df.Fjob)

df.reason = le.fit_transform(df.reason)

df.guardian = le.fit_transform(df.guardian)

df.schoolsup = le.fit_transform(df.schoolsup)

df.famsup = le.fit_transform(df.famsup)

df.paid = le.fit_transform(df.paid)

df.activities = le.fit_transform(df.activities)

df.nursery = le.fit_transform(df.nursery)

df.higher = le.fit_transform(df.higher)

df.internet = le.fit_transform(df.internet)

df.romantic = le.fit_transform(df.romantic) 

'''
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



for feat_name in object_type_features: 

    df[feat_name] = le.fit_transform(df[feat_name])
df.head(2)
X = df.drop(['G3'], axis = 1)

y = df.G3
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.metrics import recall_score, precision_score, confusion_matrix

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()



clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Train Score LR-", clf.score(X_train, y_train)*100 , "%")

print("Test Score LR-", clf.score(X_test, y_test)*100, "%")

print("Recall score", recall_score(y_test, y_pred, average='macro'))

print("Precision score", precision_score(y_test, y_pred, average='macro'))

print ("CONFUSION MATRIX", confusion_matrix(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier()



clf_dt.fit(X_train, y_train)

y_pred = clf_dt.predict(X_test)

print("Train Score LR-", clf_dt.score(X_train, y_train)*100 , "%")

print("Test Score LR-", clf_dt.score(X_test, y_test)*100, "%")

print("Recall score", recall_score(y_test, y_pred, average='macro'))

print("Precision score", precision_score(y_test, y_pred, average='macro')) 

print ("CONFUSION MATRIX", confusion_matrix(y_test, y_pred))