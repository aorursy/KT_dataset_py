import pandas as pd

import numpy as np



from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

test_df.head()
train_df['Sex'].replace({'male':0, 'female': 1}, inplace=True)

train_df['Sex'].fillna(1, inplace=True)
test_df['Sex'].replace({'male':0, 'female': 1}, inplace=True)

test_df['Sex'].fillna(1, inplace=True)
train_df['Fare'].fillna(train_df.Fare.median(), inplace=True)
test_df['Fare'].fillna(test_df.Fare.median(), inplace=True)
train_df['Pclass'].fillna(train_df.Pclass.median(), inplace=True)
test_df['Pclass'].fillna(test_df.Pclass.median(), inplace=True)
test_df.head()
train_df.corr()
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)
features = ['Sex', 'Pclass']
X_train, X_test, y_train, y_test = train_test_split(combined[features], combined.Survived, test_size=0.75, random_state=0)
logR = LogisticRegression()
guess = logR.fit(train_df[features], train_df['Survived'] )
y_test = guess.predict(test_df[features])
y_test
pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived':y_test}).set_index('PassengerId').to_csv('submission.csv')