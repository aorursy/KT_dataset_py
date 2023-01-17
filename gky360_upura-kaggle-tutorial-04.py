%matplotlib inline

import numpy as np

import pandas as pd



np.random.seed(0)
!ls ../input
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

gender_submission = pd.read_csv("../input/gender_submission.csv")
gender_submission.head()
train.head()
test.head()
data = pd.concat([train, test], sort=True)
data.head()
print(len(train), len(test), len(data))
data.isnull().sum()
data['Pclass'].value_counts()
data['Sex'].replace(['male','female'],[0, 1], inplace=True)
data['Embarked'].value_counts()
data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
age_avg = data['Age'].mean()

age_std = data['Age'].std()



data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['Family_Size'] = data['Parch'] + data['SibSp'] + 1

train['Family_Size'] = data['Family_Size'][:len(train)]

test['Family_Size'] = data['Family_Size'][len(train):]



import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x='Family_Size', data = train, hue = 'Survived')
data['IsAlone'] = 0

data.loc[data['Family_Size'] == 1, 'IsAlone'] = 1
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis = 1, inplace = True)
train = data[:len(train)]

test = data[len(train):]
y_train = train['Survived']

X_train = train.drop('Survived', axis = 1)

X_test = test.drop('Survived', axis = 1)
X_train
y_train
import lightgbm as lgb

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.1)

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
lgbm_params = {

    'learning_rate': 0.2,

    'num_leaves': 8,

    'boosting_type': 'gbdt',

    'reg_alpha': 1,

    'reg_lambda': 1,

    'objective': 'binary',

    'metric': 'auc',

}
clf = lgb.train(

    lgbm_params, lgb_train,

    valid_sets=lgb_eval,

    num_boost_round=1000,

    early_stopping_rounds=10,

)
y_pred = clf.predict(X_test)
y_pred[:20]
sub = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])

sub['Survived'] = list(map(int, y_pred))

sub.to_csv("submission.csv", index = False)

!ls .