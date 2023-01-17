

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

  



titanic_filepath = ('../input/titanic/train.csv')

titanic_data= pd.read_csv(titanic_filepath)

test_filepath = ('../input/titanic/test.csv')

test_data=pd.read_csv(test_filepath)
titanic_data.head()
features=['Sex','Fare', 'Pclass','Parch','SibSp']
x=titanic_data[features]

test_x=test_data[features]
y=titanic_data.Survived
x.head()
cleanup_nums = {"Sex":     {"male": 1, "female": 2}}

cleanup_nums2 = {"Embarked":     {"S": 1, "C": 2, "Q": 3}}

x.head()
x.replace(cleanup_nums, inplace=True)

x.head()

test_x.replace(cleanup_nums, inplace=True)

x.replace(cleanup_nums2, inplace=True)

x.head()

test_x.replace(cleanup_nums2, inplace=True)

x.head()



x=x.fillna(x.mean())

test_x=test_x.fillna(test_x.mean())

model=XGBClassifier()

model.fit(x,y)
submission_path = ('../input/titanic/gender_submission.csv')

submission= pd.read_csv(submission_path)
submission['Survived']=model.predict(test_x)







submission['PassengerId']=test_data['PassengerId']
submission.columns=['PassengerId','Survived']
submission.columns=['PassengerId','Survived']

submission.head()
submission.to_csv('Submission.csv', index=False)


