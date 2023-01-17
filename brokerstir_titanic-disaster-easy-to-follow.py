# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Load data
train = pd.read_csv("../input/titanicdatasets/train.csv")
test = pd.read_csv("../input/titanicdatasets/test.csv")

# Make a backup copy
train_copy = train.copy()
test_copy = test.copy()

# Get at an overview of the training set
print('--- Training Set ---')
print(train.columns) # Column Titles
print() # Blank Line
print(train.info()) # Detailed Column Info
print() # Blank Line
print('--- Test Set ---')
print(test.columns) # Column Titles
print() # Blank Line
print(test.info()) # Detailed Column Info
# Any results you write to the current directory are saved as output.

# Get Statistical overiew of 'Age'
print('---Training Set---')
print (train.Age.min())
print (train.Age.max())
print (train.Age.median())
print (train.Age.mean())
print() # Blank Line
print('---Test Set---')
print (test.Age.min())
print (test.Age.max())
print (test.Age.median())
print (test.Age.mean())
# Fill missing 'Age' values with the mean
train['Age']=train['Age'].fillna(np.mean(train['Age'])).astype(float)
test['Age']=test['Age'].fillna(np.mean(test['Age'])).astype(float)

# Create a new column called 'CabinBool' 
# Fill with 1 value if there was a recorded cabin, and 0 value if missing value
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

# Drop the 'Cabin' column
train.drop(labels='Cabin',inplace=True,axis=1) # inplace=True to overwrite underlying data
test.drop(labels='Cabin',inplace=True,axis=1)

# Look at a 3 row sample of training set to monitor preprocessing progress
train.sample(3)
# The Test set has no missing date in 'Embarked'
# This step will only apply to training set
# Count the number of each value that is in the 'Embarked' column
print('---Training Set---')
print(train.Embarked.value_counts())
# Fill the null values of 'Embarked' with S since it is the most frequent
#train['Embarked'] = train['Embarked'].astype(str)
train = train.fillna({"Embarked": "S"})

# Create dummy variables for the embarked column, and drop one of the dummy variables to avoid the "dummy variable trap"
train_embark=pd.get_dummies(train['Embarked'],drop_first=True,prefix='EmbarkDummy')
test_embark=pd.get_dummies(test['Embarked'],drop_first=True,prefix='EmbarkDummy')

# Have a look at the dummy variables
print('--- Training Set ---')
print(train_embark.sample())
print()
print('--- Test Set ---')
print(test_embark.sample())
# Concatenate the dummy variables onto the training and test sets
# Run Once!
train=pd.concat([train,train_embark],axis=1)
test=pd.concat([test,test_embark],axis=1)

# Drop the 'Embarked' column
train.drop(labels='Embarked',inplace=True,axis=1)
test.drop(labels='Embarked',inplace=True,axis=1)

# Look at a 3 row sample of training set to monitor preprocessing progress
train.sample(3)
# Fill the missing value in test with the mean. This step only applies to test set. 'Fare' not missing in training set.
test['Fare']=test['Fare'].fillna(np.mean(test['Fare'])).astype(float)

# Create dummy variables for gender
train_sex=pd.get_dummies(train['Sex'],drop_first=True,prefix='SexDummy')#male=1 and  female=0
test_sex=pd.get_dummies(test['Sex'],drop_first=True,prefix='SexDummy')#male=1 and  female=0

# Concatenate dummy variables on test set, and drop 'Sex' feature in same step
# Run Once!
train=pd.concat([train,train_sex],axis=1).drop(['Sex'],axis=1)
test=pd.concat([test,test_sex],axis=1).drop(['Sex'],axis=1)

# Look at a 3 row sample of each set set to monitor preprocessing progress
print (train.sample(3))
print()
print (test.sample(3))

# Use 'SibSp' and 'Parch' to create two new features called 'FamilySize' and 'IsAlone'
for row in train:
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    
for row in test:
    test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
    
for row in train:
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
    
for row in test:
    test['IsAlone'] = 0
    test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1
    
# Drop the 'SibSp' , 'Parch' , and 'FamilySize' features. Use only 'Is Alone' as binary categorical feature
train.drop(labels=['SibSp', 'Parch', 'FamilySize'],inplace=True,axis=1)
test.drop(labels=['SibSp', 'Parch', 'FamilySize'],inplace=True,axis=1)

# Look at a 3 row sample of training set to monitor preprocessing progress
train.sample(3)
# Extract title from 'Name' feature and put in new feature called 'Title'
for row in train:
    train['Title'] = train.Name.str.extract(', ([A-Za-z]+)\.', expand=False)
    
for row in test:
    test['Title'] = test.Name.str.extract(', ([A-Za-z]+)\.', expand=False)
    
# Count the values of each title in both sets
print(train.Title.value_counts())
print(test.Title.value_counts())
# Organize the lesser count titles into appropriate category
for row in train:
    train['Title'] = train['Title'].replace([ 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Sir'], 'Rare')    
    train['Title'] = train['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle'], 'Miss')

for row in test:
    test['Title'] = test['Title'].replace([ 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Sir'], 'Rare')    
    test['Title'] = test['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle'], 'Miss')

# Make dummy variables for 'Title'. and drop first dummy variable to avoid "dummy variable trap"
train_title=pd.get_dummies(train['Title'],drop_first=True,prefix='TitleDummy')
test_title=pd.get_dummies(test['Title'],drop_first=True,prefix='TitleDummy')

# Look in the sample to make sure each set has same number of dummy categories
print(train_title.sample())
print()
print(test_title.sample())
# Concatenate dummy variable and drop 'Name'
# Run Once!
train=pd.concat([train,train_title],axis=1).drop(['Name'],axis=1)
test=pd.concat([test,test_title],axis=1).drop(['Name'],axis=1)
# Reverse the order of 'Pclass' to new feature 'NewClass'
# A reversed order will give first class the highest value for the ordinal category
mapper = {1:3, 2:2, 3:1}

train['NewClass'] = train['Pclass'].map(mapper)
test['NewClass'] = test['Pclass'].map(mapper)

train["NewClass"] = train["NewClass"].astype('category')
test["NewClass"] = test["NewClass"].astype('category')

# Have a look
print(train.dtypes)
print()
print(test.dtypes)
# Put the predictors in a new variable name
train_predictors = train.drop(labels=['PassengerId', 'Survived', 'Pclass', 'Ticket', 'Title'],inplace=False,axis=1)
test_predictors = test.drop(labels=['PassengerId', 'Pclass', 'Ticket', 'Title'],inplace=False,axis=1)
# Create known target vector for training set
train_target = train.Survived
# Use traditional naming convention to split the training set and evaluate models
X = train_predictors
y = train_target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 0)
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred_gnb = gaussian.predict(X_test)
acc_gaussian = round(accuracy_score(y_pred_gnb, y_test) * 100, 2)
print(acc_gaussian)
from sklearn.metrics import confusion_matrix # Functions start with lower case
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
print(cm_gnb)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred_lr, y_test) * 100, 2)
print(acc_logreg)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
y_pred_rf = randomforest.predict(X_test)
acc_randomforest = round(accuracy_score(y_pred_rf, y_test) * 100, 2)
print(acc_randomforest)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)