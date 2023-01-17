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
import pandas as pd

train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]

print(train_df.columns)
print(train_df.columns.shape)

print(test_df.columns)
print(test_df.columns.shape)

train_df.head()
train_df.info()
print('-' * 80)
test_df.info()
train_miss = train_df.isnull().sum()
train_miss = train_miss[train_miss!=0]
#print(train_df.isnull())
test_miss = test_df.isnull().sum()
test_miss = test_miss[test_miss!=0]

print('Missing value')
print(train_miss)
print(test_miss)
print('-' * 60)
print('Missing rate')
print(train_miss.div(len(train_df)))
print(test_miss.div(len(test_df)))
train_df.describe()
test_df.describe()
train_df.describe(include='O')
test_df.describe(include='O')
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
import plotly.express as px

fig = px.histogram(train_df,x='Age', color='Survived', nbins=20)
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()

fig = px.histogram(train_df,x='Age', color='Survived', nbins=20, facet_col="Pclass")
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()
df = train_df[['Pclass', 'Embarked', 'Sex', 'Survived']].groupby(['Pclass', 'Sex', 'Embarked'], as_index=False).mean()
fig = px.scatter(df, x='Pclass', y='Survived', color='Sex', facet_col="Embarked")
fig.show()
df = train_df[['Fare', 'Sex', 'Embarked', 'Survived']].groupby(['Embarked', 'Sex', 'Survived'], as_index=False).mean()
df['Survived'] = df['Survived'].astype(str)
fig = px.bar(df, x='Sex', y='Fare', color='Survived', facet_col='Embarked', barmode='group')
fig.show()
print(train_df.shape)
print(test_df.shape)

train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId', 'Name'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)

print(train_df.shape)
print(test_df.shape)

combine = [train_df, test_df]
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

train_df.head()
fig = px.histogram(train_df, x='Age', color='Pclass', nbins=20, facet_col='Sex', 
                   barmode='overlay', marginal='box', hover_data=train_df.columns)
fig.update_traces(opacity=0.75)
fig.show()
for dataset in combine:
    for i in range(2):
        for j in range(3):
            df = dataset[(dataset['Sex'] == i) & \
                         (dataset['Pclass'] == j+1)]['Age'].dropna()
            dataset.loc[(dataset.Age.isnull()) & \
                        (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] \
            = df.median()
    dataset['Age'] = dataset['Age'].astype(int)

train_df.info()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q': 2}).astype(int)
    
train_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df.info()
test_df.info()
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis=1)
train_df.head()
test_df.head()
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(objective='binary', random_state=5)

lgbm.fit(X_train, Y_train)
Y_pred = lgbm.predict(X_test)
acc_log = round(lgbm.score(X_train, Y_train) * 100, 2)
acc_log
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': Y_pred})
#ans = pd.read_csv('/kaggle/input/gender_submission.csv')