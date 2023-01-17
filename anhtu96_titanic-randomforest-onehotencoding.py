# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/titanic'



train_df = pd.read_csv(os.path.join(path, 'train.csv'))

test_df = pd.read_csv(os.path.join(path, 'test.csv'))
train_df.shape
train_df.info()
print("Percentage of null values")

total = train_df.isnull().sum()

percent = total / len(train_df) * 100

pd.concat([total, percent], axis=1, keys=['Total', '%']).sort_values('%', ascending=False)
train_df.describe()
train_df.describe(include=['object'])
train_df.drop(columns=['PassengerId'], inplace=True)

test_id = test_df['PassengerId']

test_df.drop(columns=['PassengerId'], inplace=True)
obj_cols = [col for col in train_df.columns if train_df[col].dtype=='object']

obj_cols
train_df['Pclass'].unique()
train_df['Pclass'].value_counts(normalize=True)
train_df.groupby(['Pclass'])['Survived'].describe()
sns.countplot(train_df['Pclass'], hue=train_df['Survived'])
pd.crosstab(train_df['Survived'], train_df['Pclass'])
train_df.groupby(['Survived'])['Age'].describe()
sns.countplot(pd.qcut(train_df['Age'], 5), hue=train_df['Survived'])
sns.countplot(train_df[train_df['Age'].isnull()]['Survived'])
train_df['SibSp'].unique()
train_df['SibSp'].value_counts()
sns.countplot(train_df['SibSp'], hue=train_df['Survived'])
pd.crosstab(train_df['Survived'], train_df['SibSp'])
train_df['Parch'].value_counts()
sns.countplot(train_df['Parch'], hue=train_df['Survived'])
sns.countplot(pd.qcut(train_df['Fare'], 4), hue=train_df['Survived'])
pd.crosstab(train_df['Pclass'], pd.qcut(train_df['Fare'], 4))
train_df['Sex'].value_counts(normalize=True)
sns.countplot(train_df['Sex'], hue=train_df['Pclass'])
sns.countplot(pd.qcut(train_df['Fare'], 4), hue=train_df['Sex'])
sns.countplot(train_df['Sex'], hue=train_df['Survived'])
pd.crosstab([pd.qcut(train_df['Age'], 5), train_df['Pclass'], train_df['Sex']], train_df['Survived'])
sns.countplot(train_df['Embarked'], hue=train_df['Survived'])
sns.countplot(train_df['Embarked'], hue=train_df['Pclass'])
train_df['Name'].head()
train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split('.')[0])

test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split('.')[0])
train_df.groupby('Title')['Survived'].mean()
train_df['Ticket'].apply(lambda x: str(x))
ticket_len = train_df['Ticket'].apply(lambda x: len(x.split(' ')))
sns.countplot(ticket_len, hue=train_df['Pclass'])
# impute age

age_group = train_df.groupby(['Title', 'Pclass'])['Age']

train_df['Age'] = age_group.apply(lambda x: x.fillna(x.mean()))

test_df['Age'] = age_group.apply(lambda x: x.fillna(x.mean()))
# check if there're any null values in SibSp and Parch for test set

print(test_df['SibSp'].isnull().sum(), test_df['Parch'].isnull().sum())
# number of relatives

train_df['Relatives_num'] = train_df['SibSp'] + train_df['Parch']

test_df['Relatives_num'] = test_df['SibSp'] + test_df['Parch']
# impute Embarked

train_df['Embarked'].fillna('S', inplace=True)

test_df['Embarked'].fillna('S', inplace=True)
# impute Fare

test_df['Fare'].fillna(train_df['Fare'].mean(), inplace=True)
drop_cols = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']

X_train = train_df.copy()

y_train = train_df['Survived']

X_test = test_df.copy()



X_train.drop(columns=drop_cols + ['Survived'], inplace=True)

X_test.drop(columns=drop_cols, inplace=True)
# One-hot encoding



categorical_cols = ['Pclass', 'Sex', 'Embarked', 'Title']

categorical_cols



for col in categorical_cols:

    col_vals = [vals for vals in X_train[col].unique()]

    onehot_cols = [col + '_' + str(s) for s in col_vals]

    dummies_train = pd.get_dummies(X_train[col], prefix=col)

    dummies_test = pd.get_dummies(X_test[col], prefix=col)

    missing_cols = set(dummies_train.columns) - set(dummies_test.columns)

    for missing_col in missing_cols:

        dummies_test[missing_col] = 0

    X_train = pd.concat([X_train, dummies_train], axis=1)

    X_test = pd.concat([X_test, dummies_test[onehot_cols]], axis=1)

    X_train.drop(columns=col, inplace=True)

    X_test.drop(columns=col, inplace=True)

    

X_test = X_test[X_train.columns]
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1, n_jobs=-1, oob_score=True)

skf = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)

params = {'n_estimators': [200, 500],

          'max_depth': [15, 30, 50],

         'min_samples_leaf': [1, 2, 5, 10]}

gcv = GridSearchCV(rf, params, scoring='accuracy', n_jobs=-1, cv=skf, verbose=1)

gcv.fit(X_train, y_train)
gcv.best_score_
gcv.best_params_
gcv.best_estimator_.feature_importances_
rf = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_leaf=5, n_jobs=-1, oob_score=True, random_state=1)

rf.fit(X_train, y_train)

rf.oob_score_
pred_test = rf.predict(X_test)
output = pd.DataFrame({'PassengerId': test_id, 'Survived': pred_test})
output.to_csv('submission.csv', index=False)