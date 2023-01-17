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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

test_ds = pd.read_csv('/kaggle/input/titanic/test.csv')

gen_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
print (dataset)
print (test_ds)
print (gen_sub)
dataset.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',

        inplace=True)

test_ds.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',

        inplace=True)

gen_sub.drop('PassengerId', axis=1, inplace=True)
print (dataset)
dataset.Sex.value_counts()
test_ds.Sex.value_counts()
dataset[["Sex", "Survived"]].groupby(dataset["Sex"], as_index = False).mean()
X = dataset.drop('Survived', axis='columns')

y = dataset.Survived
X.columns[X.isna().any()]
X.isna().sum()
test_ds.columns[test_ds.isna().any()]
test_ds.isna().sum()
X.Age = X.Age.fillna(X.Age.median())
test_ds = test_ds.fillna({'Age':test_ds.Age.median(),

                          'Fare':test_ds.Fare.median()})
X.columns[X.isna().any()]
test_ds.columns[test_ds.isna().any()]
print (X)
X.describe()
dummies = pd.get_dummies(X.Sex)
dummies1 = pd.get_dummies(test_ds.Sex)
X = pd.concat([X,dummies],axis='columns')
test_ds = pd.concat([test_ds,dummies1],axis='columns')
X.drop(['Sex','female'],axis='columns',inplace=True)
test_ds.drop(['Sex','female'],axis='columns',inplace=True)
corr = dataset.corr()
top_corr_features = corr.index
plt.figure(figsize=(20,20))

g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

test_ds = sc.transform(test_ds)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X, y)
y_pred = classifier.predict(test_ds)
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(confusion_matrix(gen_sub, y_pred))
print (accuracy_score(gen_sub,y_pred))
ac = accuracy_score(gen_sub,y_pred)
print ("{0:.0%}".format(ac))