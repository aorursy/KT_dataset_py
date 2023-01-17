import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fastai.imports import *
from fastai.structured import *
train_csv = pd.read_csv('titanic/train.csv')
train_csv = train_csv.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Fare'])
train_csv.head()
train_csv.describe()
train_csv['Cabin'][~train_csv['Cabin'].isnull()] = 1  # not nan
train_csv['Cabin'][train_csv['Cabin'].isnull()] = 0   
train_csv['Age'] = train_csv['Age'].fillna(train_csv['Age'].mean())
train_cats(train_csv)
train_csv.Sex = train_csv.Sex.cat.codes
train_csv.Embarked = train_csv.Embarked.cat.codes
model = RandomForestClassifier(n_jobs = -1, max_depth = 5)
param_df, target, na_cols = proc_df(train_csv, 'Survived')
model.fit(param_df, target)
model.score(param_df, target)
test_df = pd.read_csv('titanic/test.csv')
passenger_list = test_df.PassengerId.tolist()
test_df = test_df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Fare'])
test_df.head()
test_df['Cabin'][~test_df['Cabin'].isnull()] = 1  # not nan
test_df['Cabin'][test_df['Cabin'].isnull()] = 0   
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())
train_cats(test_df)
test_df.Sex = test_df.Sex.cat.codes
test_df.Embarked = test_df.Embarked.cat.codes
test_df.head()
survived_pred = model.predict(test_df).tolist()
final = {'PassengerId': passenger_list, 'Survived': survived_pred}
print("Length passengers: " + str(len(passenger_list)) + "\nLength survived predictions: " + str(len(survived_pred)))
submission = pd.DataFrame(final)
submission.to_csv('titanic/submission.csv', index = False)