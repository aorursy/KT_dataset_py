#import libraries

import pandas as pd

from sklearn.ensemble import RandomForestClassifier



#Get data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# First 5 rows

train.head()
train.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

test.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

train.head()
one_hot_train = pd.get_dummies(train)

one_hot_test = pd.get_dummies(test)



# First five rows from train dataset

one_hot_train.head()
# First five rows from test dataset

one_hot_test.head()
# Visualize the null values (train)

one_hot_train.isnull().sum().sort_values(ascending=False)
# Fill the null Age values with the mean of all ages

one_hot_train['Age'].fillna(one_hot_train['Age'].mean(), inplace=True)

one_hot_test['Age'].fillna(one_hot_test['Age'].mean(), inplace=True)

one_hot_train.isnull().sum()
# Visualize the null values (test)

one_hot_test.isnull().sum().sort_values(ascending=False)
# Fill the null Fare values with the mean of all Fares

one_hot_test['Fare'].fillna(one_hot_test['Fare'].mean(), inplace=True)

one_hot_test.isnull().sum().sort_values(ascending=False)
# Creating the feature and the target

feature = one_hot_train.drop('Survived', axis=1)

target = one_hot_train['Survived']



# Model creation

rf = RandomForestClassifier(random_state=1, criterion='gini', max_depth=10, n_estimators=50, n_jobs=-1)

rf.fit(feature, target)
# Verifying score

rf.score(feature, target)
# Generate a DataFrame with Padas with 'PassengerId' and 'Survived' colunms

submission = pd.DataFrame()

submission['PassengerId'] = one_hot_test['PassengerId']

submission['Survived'] = rf.predict(one_hot_test)



# Generate the CSV file with 'to_csv' from Pandas

submission.to_csv('submission.csv', index=False)