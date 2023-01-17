import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
train.describe()
train.shape
train.isna().sum()
columns = train.columns
plt.hist(train['Age'])



plt.show()
Pclass_y = list(train['Pclass'].value_counts())

Pclass_x = list((train['Pclass']).unique())
plt.bar(Pclass_x, Pclass_y)

plt.show()
train.head()
# Function to create povit tables to determine how the elements of the DF relate the the survival rate



def pivot(x):

    return(pd.pivot_table(train,index=[x], values=['Survived'], aggfunc = np.sum))
pivot('Sex')
pivot('Pclass')
pivot('Parch')
pivot('Embarked')
train.isna().sum()
pivot('Cabin')
train.head()
train = train.drop(columns=['PassengerId', 'Name', 'Ticket'])
train['Age'].fillna(train['Age'].median(), inplace=True)
train.isna().sum()
train.head()
train['Cabin'].unique()
train['Cabin'].fillna('other', inplace=True)
train['Cabin'] = train['Cabin'].apply(lambda x: x[0])
sex_dummies = pd.get_dummies(train['Sex'], drop_first=True)



embarked_dummies = pd.get_dummies(train['Embarked'], drop_first=True)



cabin_dummies = pd.get_dummies(train['Cabin'], drop_first=True)
y = train['Survived']



final_data = train.drop(columns = ['Survived', 'Sex', 'Embarked', 'Cabin']) 

final_data_list = [final_data, sex_dummies, embarked_dummies, cabin_dummies]



X = pd.concat(final_data_list, axis='columns')
# splitting the data (train/test)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train
# Logistic Regression

# Decision Tree 

# Random Forest

# Support Vector Machine 

# KNN
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



clf_lr = LogisticRegression(max_iter=200)



clf_lr.fit(X_train, y_train)



clf_lr_pred = clf_lr.predict(X_test)



f1_score(y_test, clf_lr_pred)
from sklearn import tree



clf_tree = tree.DecisionTreeClassifier()



clf_tree = clf_tree.fit(X_train, y_train)



clf_tree_pred = clf_tree.predict(X_test)



f1_score(y_test, clf_tree_pred)
from sklearn.ensemble import RandomForestClassifier 



clf_rfc = RandomForestClassifier()



clf_rfc = clf_rfc.fit(X_train, y_train)



clf_rfc_pred = clf_rfc.predict(X_test)



f1_score(y_test, clf_rfc_pred)
# test Data 

test = pd.read_csv('test.csv')
test.head()
test.isna().sum()
test['Age'].fillna(test['Age'].median(), inplace=True)

test['Fare'].fillna(test['Fare'].median(), inplace=True)

test['Cabin'].fillna('other', inplace=True)
test['Cabin'] = test['Cabin'].apply(lambda x: x[0])
sex_dummies = pd.get_dummies(test['Sex'], drop_first=True)



embarked_dummies = pd.get_dummies(test['Embarked'], drop_first=True)



cabin_dummies = pd.get_dummies(test['Cabin'])

test.head()
test_data = test.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked'])
data_sub = pd.concat([test_data, sex_dummies, embarked_dummies, cabin_dummies], axis='columns')
y_sub = clf_lr.predict(data_sub)
submission = pd.DataFrame(columns = ['PassengerId', 'Survived'])
submission['PassengerId'] = test['PassengerId']

submission['Survived'] = y_sub
submission.to_csv('submission', index=False)