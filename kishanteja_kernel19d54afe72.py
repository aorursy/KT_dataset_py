# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('/kaggle/input/titanic/train.csv')

X_test = pd.read_csv('/kaggle/input/titanic/test.csv')
X_train.info()
X_test.info()
feature_columns = ['Pclass','Sex','Age','SibSp','Parch','Embarked']

y_train = X_train['Survived']

y_pass = X_test['PassengerId']

X_train = X_train[feature_columns]

X_test = X_test[feature_columns]
common_age1 = X_train['Age'].mean()

common_age2 = X_test['Age'].mean()



X_train['Age'] = X_train['Age'].fillna(common_age1)

X_test['Age'] = X_test['Age'].fillna(common_age2)
common = "S"



X_train['Embarked'] = X_train['Embarked'].fillna(common)
data = [X_train,X_test]
dict1 = {"S":0 ,"C":1 ,"Q":2}



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(dict1)

    

dict2 = {"male":0 , "female":1}



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(dict2)
for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
dectreclf = XGBClassifier(n_estimators = 1000,learning_rate = 0.05,n_jobs =4)

dectreclf.fit(X_train,y_train)



y_pred = dectreclf.predict(X_test)



accuracy = round(dectreclf.score(X_train,y_train)*100,2)

print(round(accuracy,2,),"%")
submission = pd.DataFrame({

        "PassengerId": y_pass,

        "Survived":y_pred

    })

submission.to_csv('submission.csv', index=False)