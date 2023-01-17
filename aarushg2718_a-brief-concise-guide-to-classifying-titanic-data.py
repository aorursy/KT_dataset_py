import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

import xgboost as xgb
raw_train_data = pd.read_csv('../input/titanic/train.csv')
raw_test_data = pd.read_csv('../input/titanic/test.csv')
raw_train_data.info()
raw_train_data.describe()
raw_train_data.head()
print("Number of missing values:")
print(raw_train_data.isnull().sum())
def clean_data(data, drop_id=True):
    
    data['HasCabin'] = ~data['Cabin'].isnull()
   
    data['Age'].fillna(data['Age'].median(), inplace=True)
    
    data.dropna(subset=['Embarked'], how='any', inplace=True)
    
    data['IsMale'] = data['Sex'].apply(lambda x : x == "male")
    
    data['Title'] = data['Name'].apply(lambda x : x.split(',')[-1].split(' ')[1][:-1])
    titles = (data['Title'].value_counts() > 10)
    data['Title'] = data['Title'].apply(lambda x: x if titles[x] else 'Misc')
    data = data.join(pd.get_dummies(data['Title']))
    
    data = data.join(pd.get_dummies(data['Embarked']))
    
    data.drop(columns=['Ticket', 'Name', 'Title', 'Sex', 'Embarked', 'Cabin'], inplace=True)
    if drop_id:
        data.drop(columns=['PassengerId'], inplace=True)
    
    return data
    
train_data = clean_data(raw_train_data)
test_data = clean_data(raw_test_data, drop_id=False)

x_train = train_data.drop(columns=['Survived'])
y_train = train_data['Survived']
sns.heatmap(train_data.corr(), annot=True)
mpl.rcParams['figure.figsize'] = 26, 26
plt.show()
xgbc = xgb.XGBClassifier()
xgbc.fit(x_train, y_train)
print(f"Cross-val score: {cross_val_score(xgbc, x_train, y_train, cv=5).mean()}")
1.0 * y_train.value_counts()[0] / len(y_train)
submissions_path = "/kaggle/working/submission.csv"
predictions = pd.Series(xgbc.predict(test_data.drop(columns=['PassengerId'])))

predictions = pd.concat([test_data['PassengerId'], predictions], axis=1)
predictions.columns = ['PassengerId', 'Survived']

predictions.to_csv(submissions_path)