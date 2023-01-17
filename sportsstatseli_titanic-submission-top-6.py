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
valid = pd.read_csv('/kaggle/input/titanic/test.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test_subm = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train.drop(['Name', 'Ticket'],inplace=True, axis=1)

train
train.drop(["Cabin"], inplace=True, axis=1)
train["Family"] = train['SibSp'] + train['Parch']+1

train['Fare_person'] = train['Fare']/train['Family']

train['Safety'] = train['Pclass']*10 + train['Age']

train
valid.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'],inplace=True, axis=1)
valid["Family"] = valid['SibSp'] + valid['Parch']+1

valid['Fare_person'] = valid['Fare']/valid['Family']

valid['Safety'] = valid['Pclass']*10 + valid['Age']

sex = {'male': 1, 'female':0}

train.replace({'Sex': sex}, inplace=True)

valid.replace({'Sex': sex}, inplace=True)
e_map = {'S': 1, 'C':2,'S':3, 'Q':4}

train.replace({'Embarked': e_map}, inplace=True)

valid.replace({'Embarked': e_map}, inplace=True)
train
from sklearn import preprocessing



feats = train[['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked', 'Family','Fare_person','Safety']]



scaler = preprocessing.StandardScaler()

all_features_scaled = scaler.fit_transform(feats)

all_features_scaleddf = pd.DataFrame(all_features_scaled)
from sklearn import preprocessing



feats2 = valid[['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked', 'Family','Fare_person','Safety']]



scaler = preprocessing.StandardScaler()

all_features_scaled2 = scaler.fit_transform(feats2)

all_features_scaleddf2 = pd.DataFrame(all_features_scaled2)
data = train[['PassengerId','Survived']].join(all_features_scaleddf)

data.columns = ['PassengerId','Survived','Pclass','Age','SibSp','Parch','Fare','Sex','Embarked', 'Family','Fare_person','Safety']



data2 = all_features_scaleddf2

data2.columns = ['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked', 'Family','Fare_person','Safety']
data = data.fillna(data.median())

data2 = data2.fillna(data2.median())
data
from sklearn.model_selection import train_test_split

import xgboost as xgb



X_train, X_test, y_train, y_test = train_test_split(data[['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked', 'Family','Fare_person','Safety']], data[["Survived"]], test_size=0.25, random_state=0)
from sklearn.ensemble import RandomForestClassifier

RF= RandomForestClassifier(n_estimators=100,random_state=22)
RF.fit(X_train,y_train)
y_pred = RF.predict(data2)
print("Train Score:",RF.score(X_train, y_train))
test_subm.drop(["Survived"], inplace=True, axis = 1)

test_subm.insert(1, "Survived", y_pred)

test_subm
test_subm.to_csv('sub7.csv', index=False)