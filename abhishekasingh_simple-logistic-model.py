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
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train =  pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
train.columns = train.columns.str.lower()
train.info()
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data_copy = test.copy()
test.head()
test.columns = test.columns.str.lower()
test.info()
test_data_copy.info()
train.isnull().sum()
plt.figure(figsize=(14,8))

plt.title('Missing values in Training',{'fontsize': 25},pad = 20)

sns.heatmap(train.isnull(), cmap='viridis')
plt.figure(figsize=(14,8))

plt.title('Age distribution in training data',{'fontsize': 25},pad = 20)

sns.boxplot(x="pclass", y="age", data=train,palette='rainbow')
train.groupby("pclass")['age'].median()
train['age'] = train['age'].fillna(train.groupby(["sex","pclass"])['age'].transform('median'))
plt.figure(figsize=(14,8))

sns.heatmap(train.isnull(),cmap='viridis')
train.drop('cabin',axis=1,inplace=True)
train.dropna(inplace=True)
plt.figure(figsize=(14,8))

sns.heatmap(train.isnull(), cmap='viridis')
test.isnull().sum()
plt.figure(figsize=(14,8))

plt.title('Missing values in test data',{'fontsize': 25},pad = 20)

sns.heatmap(test.isnull(), cmap='viridis')
plt.figure(figsize=(14,8))

plt.title('Age distribution in test data',{'fontsize': 25},pad = 20)

sns.boxplot(x="pclass", y="age", data=test, palette='rainbow')
test.groupby("pclass")['age'].median()
test['age'] = test['age'].fillna(test.groupby(["sex","pclass"])['age'].transform('median'))
test.groupby("pclass")['fare'].median()
test['fare'] = test['fare'].fillna(test.groupby(["sex","pclass"])['fare'].transform('median'))
test.drop('cabin',axis=1,inplace=True)
test.dropna(inplace=True)
plt.figure(figsize=(14,8))

plt.title('Missing values in test data',{'fontsize': 25},pad = 20)

sns.heatmap(test.isnull(), cmap='viridis')
train.info()
train_data_sex = pd.get_dummies(train['sex'],drop_first=True)

test_data_sex = pd.get_dummies(test['sex'],drop_first=True)
train_data_embark = pd.get_dummies(train['embarked'],drop_first=True)

test_data_embark = pd.get_dummies(test['embarked'],drop_first=True)
train.drop(['sex','embarked','name','ticket','passengerid'],axis=1,inplace=True)

test.drop(['sex','embarked','name','ticket','passengerid'],axis=1,inplace=True)
train.info()
test.info()
train = pd.concat([train, train_data_sex, train_data_embark],axis=1)
test = pd.concat([test, test_data_sex, test_data_embark],axis=1)
X_train = train.drop("survived", axis=1)

y_train = train["survived"]
X_train.shape, y_train.shape
X_test = test
X_test.shape
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(C=0.001)

logmodel.fit(X_train,y_train)
y_pred = logmodel.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test_data_copy["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('titanic.csv', index=False)
print('My First Kaggle Submission')
import xgboost as xgb
xg_cls = xgb.XGBClassifier()
xg_cls.fit(X_train, y_train)
xg_preds = xg_cls.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test_data_copy["PassengerId"],

        "Survived": xg_preds

    })

submission.to_csv('xgmodel.csv', index=False)