# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import BaseEstimator,TransformerMixin



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
train_data.info()
train_data.describe()
%matplotlib inline
sns.countplot(train_data['Survived'],data=train_data,hue='Sex')

sns.countplot(train_data['Survived'],data=train_data,hue='Pclass')
train_data.Cabin.fillna('U',inplace=True)

train_data['Cabin'] = train_data['Cabin'].map(lambda c: c[0])
sns.countplot(train_data['Survived'],data=train_data,hue='Cabin')
a_imputer = SimpleImputer(strategy='constant',fill_value='unknown')

a_imputer.fit(train_data[['Embarked']])

train_data['Embarked'] = a_imputer.transform(train_data[['Embarked']])
num_imputer = SimpleImputer(strategy='median')

num_imputer.fit(train_data[['Age']])

train_data['Age'] = num_imputer.transform(train_data[['Age']])
train_data.head()
train_data.info()
train_data['Male'] = train_data['Sex'].map({'male':1,'female':0})
onehotcols = ['Embarked','Cabin']

onehot = OneHotEncoder(sparse=False,handle_unknown='ignore')

for onecol in onehotcols:

    train_encoded = pd.DataFrame(onehot.fit_transform(train_data[[onecol]]))

    train_encoded.columns = onehot.get_feature_names([onecol])

    train_data.drop(onecol ,axis=1, inplace=True)

    train_data= pd.concat([train_data, train_encoded ], axis=1)
train_data.info()
train_data.head()
id_pass = train_data.PassengerId

y = train_data.Survived
train_data.drop(['Name','Ticket','Sex','PassengerId','Survived'],axis=1,inplace=True)
train_data.head()
scaler=StandardScaler()

scaler.fit(train_data)

scaled_features=scaler.transform(train_data)

train_data=pd.DataFrame(scaled_features,columns=train_data.columns)
train_data.head()
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(train_data,y,test_size=0.2,random_state=42)
models = []

import lightgbm as lgb

clf = lgb.LGBMClassifier(n_estimators=70,learning_rate=0.05,max_depth=10,num_leaves=10,random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)
#Accuracy

from sklearn.metrics import accuracy_score

accuracylg=accuracy_score(y_pred,y_valid)

accuracylg
from sklearn.metrics import classification_report as cr

print(cr(y_valid,y_pred))
from xgboost import XGBClassifier

my_model1 = XGBClassifier(n_estimators=70,learning_rate=0.05,max_depth=7,random_state=42)

my_model1.fit(X_train,y_train)
y_pred1=my_model1.predict(X_valid)
accuracyxg=accuracy_score(y_pred1,y_valid)

accuracyxg
print(cr(y_valid,y_pred1))
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.info()
test_data.Cabin.fillna('U',inplace=True)

test_data['Cabin'] = test_data['Cabin'].map(lambda c: c[0])

a_imputer.fit(test_data[['Embarked']])

test_data['Embarked'] = a_imputer.transform(test_data[['Embarked']])

num_imputer.fit(test_data[['Age']])

test_data['Age'] = num_imputer.transform(test_data[['Age']])

num_imputer.fit(test_data[['Fare']])

test_data['Fare'] = num_imputer.transform(test_data[['Fare']])
test_data.info()
test_data['Male'] = test_data['Sex'].map({'male':1,'female':0})

for onecol in onehotcols:

    test_encoded = pd.DataFrame(onehot.fit_transform(test_data[[onecol]]))

    test_encoded.columns = onehot.get_feature_names([onecol])

    test_data.drop(onecol ,axis=1, inplace=True)

    test_data= pd.concat([test_data, test_encoded ], axis=1)
test_data.head()
id_pass_test = test_data.PassengerId
test_data.drop(['Name','Ticket','Sex','PassengerId'],axis=1,inplace=True)
test_data.head()
idx = 9

new_col = 0.0  # can be a list, a Series, an array or a scalar   

test_data.insert(loc=idx, column='Embarked_unknown', value=new_col)
idx = 17

new_col = 0.0  # can be a list, a Series, an array or a scalar   

test_data.insert(loc=idx, column='Cabin_T', value=new_col)
scaler.fit(test_data)

scaled_features1=scaler.transform(test_data)

test_data=pd.DataFrame(scaled_features1,columns=test_data.columns)
test_data.head()
y_pred_test=my_model1.predict(test_data)
y_pred_test
predi = pd.DataFrame(y_pred_test, columns=['predictions'])

id_df = pd.DataFrame(id_pass_test, columns=['PassengerId'])
Ser_data = [id_df["PassengerId"], predi["predictions"]]

col_header = ["PassengerId", "Survived"]

final_data = pd. concat(Ser_data, axis=1, keys=col_header)
final_data.head()
final_data.to_csv("/kaggle/working/my_sub.csv",index=False)