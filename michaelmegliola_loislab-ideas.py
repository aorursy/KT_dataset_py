import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train
# this code says: extract any substring that starts with a uppercase letter and ends with a '.'

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.')

print('here are the unique set of titles: ',train['Title'].unique())
# this function splits a string into a list, then returns the length of the list

# the splitting uses a the space character ' ' as a delimiter, so 'A B C' would

# become: ['A','B','C'], and len(['A','B','C']) = 3

def count_number_of_tokens(s):

    return len(s.split(' '))



# you can apply a function to a series (in this case, to make a new series)

train['NameCount'] = train['Name'].apply(count_number_of_tokens)

train[['Name','Title','NameCount']]
train["Age"] = train["Age"].fillna(-1)

bins = [-2, 0, 8, 16, 24, 36, 60, np.inf]

labels = ['Unknown', 'Child', 'Tween', 'YA', 'Adult', 'Boomer', 'Oldster']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

train[['Age','AgeGroup']]
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



# dependent variable... the thing we are trying to predict

y = train['Survived']



# independent variables...

features = ['Sex','Embarked','AgeGroup','NameCount','Pclass', 'Fare']

X = pd.get_dummies(train[features])

X
model = RandomForestClassifier(n_estimators=100)

model.fit(X, y)

predictions = model.predict(X)

print(np.shape(X),X.columns)

print('Accuracy is ',metrics.accuracy_score(predictions,y))
# get the test data set

test = pd.read_csv('/kaggle/input/titanic/test.csv')



# add our data features...

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.')

test['NameCount'] = test['Name'].apply(count_number_of_tokens)

test["Age"] = test["Age"].fillna(-1)

bins = [-2, 0, 8, 16, 24, 36, 60, np.inf]

labels = ['Unknown', 'Child', 'Tween', 'YA', 'Adult', 'Boomer', 'Oldster']

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

test[['Age','AgeGroup']]



# independent variables for which our model is already fit...

features = ['Sex','Embarked','AgeGroup','NameCount','Pclass','Fare']

X_test = pd.get_dummies(test[features])



# wait! some fields are missing... that will confuse the model.

# which are the missing fields?

missing_fields = set(X.columns) - set(X_test.columns)

print('missing fields',missing_fields)

# populate the missing fields with zeros

for field in missing_fields:

    X_test[field] = 0

# make sure the columns are in the same order as the orginal

# set of independent variables X:

X_test = X_test[X.columns]

# check that alignment is correct

for s,t in zip(X.columns, X_test.columns):

    print(s,t,s==t)
X_test
median = X_test.median()['Fare']

print('median =', median)

X_test['Fare'] = X_test['Fare'].fillna(median)
Y_test = model.predict(X_test)

lois_lab_result = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':Y_test})

lois_lab_result.to_csv('lois_lab_submission_2.csv', index=False)

print(np.shape(lois_lab_result))

lois_lab_result