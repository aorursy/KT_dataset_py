# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
app_train = pd.read_csv("../input/train.csv", na_values=['NaN'])
app_test = pd.read_csv('../input/test.csv', na_values=['NaN'])

app_train.head()
app_test.isnull().sum()
print("app data shape ",app_train.shape)
print("app data shape ",len(app_train))
missing_values = app_train.isnull().sum()/len(app_train)*100
print(missing_values)
#drop columns where missing_values are atleast 60% of data
thresh = len(app_train)*.4
app_train_dropped = app_train.dropna(thresh=thresh,axis=1,inplace=True)
app_train.head()
# mean_age = app_train['Age'].mean()
# app_train['Age'] = app_train['Age'].fillna(mean_age)
# app_train.loc[app_train['Pclass']==3,['Pclass']] = 'L'
# app_train.loc[app_train['Pclass']==2,['Pclass']] = 'M'
# app_train.loc[app_train['Pclass']==1,['Pclass']] = 'H'

# app_test.loc[app_test['Pclass']==3,['Pclass']] = 'L'
# app_test.loc[app_test['Pclass']==2,['Pclass']] = 'M'
# app_test.loc[app_test['Pclass']==1,['Pclass']] = 'H'
#one-hot encoder for multi variate categorical columns

ids = app_train['PassengerId']
test_ids = app_test['PassengerId'] 
app_train = app_train.drop(columns=['Name','Ticket','PassengerId'])
labels = app_train['Survived']
app_train,app_test = app_train.align(app_test,join='inner', axis = 1)
app_train['Survived'] = labels
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

app_train.head()
app_train.corr()['Survived']
app_train['Family_size'] = app_train['SibSp']+app_train['Parch']
app_test['Family_size'] = app_test['SibSp']+app_test['Parch']

# app_train = app_train.drop(columns=['SibSp','Parch'])
# app_test = app_test.drop(columns=['SibSp','Parch'])

app_train['Family_size'].plot(kind='kde')
survived_fare = app_train.loc[app_train['Survived']==1,['Fare']]
unsurvived_fare = app_train.loc[app_train['Survived']==0,['Fare']]
print(np.mean(unsurvived_fare.values))
print(np.mean(survived_fare.values))
poly_features = app_train.loc[:,['Age']]
poly_test_features = app_test.loc[:,['Age']]
# from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(strategy='median')
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')
poly_features = imputer.fit_transform(poly_features)
poly_test_features = imputer.transform(poly_test_features)

app_train['Age'] = poly_features
app_test['Age'] = poly_test_features

# Fill in missing values of Fare with the average Fare
if len(app_test[app_test['Fare'].isnull()] > 0):
    avg_fare = app_test['Fare'].mean()
    app_test.replace({ None: avg_fare }, inplace=True)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
train_features = app_train.drop(columns=['Survived'])
# Fit the model to our training data
clf = clf.fit(train_features.values, labels)

# Get the test data features, skipping the first column 'PassengerId'
test_x = app_test.values
test_x = test_x[:,:]

# Predict the Survival values for the test data
test_y = clf.predict(test_x)

submit = pd.DataFrame(test_ids,columns=['PassengerId'])
submit['Survived'] = test_y
submit.head()

# # Save the submission to a csv file
submit.to_csv('predicted_survival_rate_new.csv', index = False)

