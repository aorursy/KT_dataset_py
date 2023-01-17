import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
# getting the training and test data information

# train_data.info()

# test_data.info()
# dropping useless columns
useless_cols = ['Name', 'Ticket', 'Cabin']
train_data = train_data.drop(useless_cols, axis=1)
test_data = test_data.drop(useless_cols, axis=1)
# test_data.info()
# train_data.info()
# dummy variables
dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(train_data[col]))
    
titanic_dummies = pd.concat(dummies, axis=1)

# concatenate to the original
train_data = pd.concat((train_data,titanic_dummies), axis=1)

dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(test_data[col]))
    
titanic_dummies = pd.concat(dummies, axis=1)

test_data = pd.concat((test_data,titanic_dummies), axis=1)


# drop the columns
train_data = train_data.drop(['Pclass', 'Sex', 'Embarked'], axis=1)

test_data = test_data.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
# train_data.info()

# test_data.info()
train_data['Age'] = train_data['Age'].interpolate()

test_data['Age'] = test_data['Age'].interpolate()
test_data['Fare'] = test_data['Fare'].interpolate()
# train_data.info()

# test_data.info()
#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

y = train_data["Survived"]

X =  train_data.loc[:, train_data.columns != 'Survived']
X_test = test_data.values

#Train the model using the training sets
clf.fit(X, y)

predictions = clf.predict(X_test)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
