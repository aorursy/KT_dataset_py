# This Python 3 environment comes with many helpful analytics libraries installed

# As input data, You need download dataset you want 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from xgboost import XGBClassifier



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
# train.csv contains data

train_csv = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv")

train_csv.tail(5)
# Age, Cabin and Embarked have Missing data

train_csv.info()
## 1.1. Delete irrelevant Features ('PassengerId', 'Name', 'Ticket') ##



train_csv.drop('PassengerId', axis=1, inplace=True )

train_csv.drop('Name', axis=1, inplace=True )

train_csv.drop('Ticket', axis=1, inplace=True )

train_csv.info()
# test.csv is test data to check the accuracy of the model created

test_csv = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

test_csv.tail(5)
# Need to consider mean of training data and test data ('Age', 'Fare')

n_test_nna_age = test_csv['Age'].notnull().sum()

n_train_nna_age = train_csv['Age'].notnull().sum()



n_test_nna_fare = test_csv['Fare'].notnull().sum()

n_train_nna_fare = train_csv['Fare'].notnull().sum()



weighted_mean_age = (n_train_nna_age * train_csv['Age'].mean() + n_test_nna_age * test_csv['Age'].mean()) / (n_test_nna_age + n_train_nna_age)

weighted_mean_fare = (n_train_nna_fare * train_csv['Fare'].mean() + n_test_nna_fare * test_csv['Fare'].mean()) / (n_test_nna_fare + n_train_nna_fare)
## 1.2. Missing Value Preprocessing ('Age', 'Embarked', 'Cabin') ##

# 'Age', 'Embarked' Preprocessing

# The number of missing values of 'Age' is relatively few,(177) So missing values can be filled with mean.

train_csv['Age'].fillna(weighted_mean_age, inplace=True)



# The number of missing values of 'Embarked' is alse few,(2) So missing values can be filled with the most(S).

print('Distribution of \'Embarked\' :\n',train_csv['Embarked'].value_counts())

train_csv['Embarked'].fillna('S',inplace=True)
# 'Cabin' Preprocessing

# When passenger has NaN in 'Cabin', passenger has more chance to Survive



# False : NaN / True : A?? ~ G??

train_csv_isnull = train_csv.isnull()

train_csv['Cabin'] = train_csv_isnull['Cabin']

sns.barplot(x='Cabin', y = 'Survived', data=train_csv)
## 1.3. Replace Value ('Sex', 'Embarked', 'Cabin') ## 

# 'Cabin' Preprocessing

train_csv.replace({'Cabin': True}, 1, inplace=True)

train_csv.replace({'Cabin': False}, -1, inplace=True)



# 'Sex' Preprocessing

sns.barplot(x='Sex', y = 'Survived', data=train_csv)

train_csv.replace({'Sex': 'female'}, 1, inplace=True)

train_csv.replace({'Sex': 'male'}, -1, inplace=True)
# 'Embarked' Preprocessing

sns.barplot(x='Embarked', y = 'Survived', data=train_csv)

train_csv.replace({'Embarked': 'S'}, -1, inplace=True)

train_csv.replace({'Embarked': 'C'}, 1, inplace=True)

train_csv.replace({'Embarked': 'Q'}, 0, inplace=True)
## 1.4. Data Normalization('Pclass', 'Age', 'SibSp', 'Parch', 'Fare') ##

# Pclass : 1,2,3 -> 1,0,-1

sns.barplot(x='Pclass', y = 'Survived', data=train_csv)

train_csv['Pclass'] = train_csv['Pclass'].apply(lambda x:-x+2)
# 'SibSp' : 8 5 4 3 0 2 1 -> -1, -0.66, -0.33, 0, 0.33, 0.66, 1

def SibSp_prep(x):

    

    if x == 8:

        out = -1

    elif x == 5:

        out = -0.66

    elif x == 4:

        out = -0.33

    elif x == 3:

        out = 0

    elif x == 0:

        out = 0.33

    elif x == 2:

        out = 0.66

    else:

        out = 1

    return out



print(train_csv['SibSp'].max())

sns.barplot(x='SibSp', y = 'Survived', data=train_csv)

train_csv['SibSp'] = train_csv['SibSp'].apply(SibSp_prep)
# 'Parch' : 6 4 5 0 2 1 3 -> -1, -0.66, -0.33, 0, 0.33, 0.66, 1

def Parch_prep(x):

    

    if x == 6:

        out = -1

    elif x == 4:

        out = -0.66

    elif x == 5:

        out = -0.33

    elif x == 0:

        out = 0

    elif x == 2:

        out = 0.33

    elif x == 1:

        out = 0.66

    else:

        out = 1

    return out



print(train_csv['Parch'].max())

sns.barplot(x='Parch', y = 'Survived', data=train_csv)

train_csv['Parch'] = train_csv['Parch'].apply(Parch_prep)
# test.csv is test data to check the accuracy of the model created

test_csv = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

test_csv.tail(5)



num_test = test_csv.shape[0]

num_train = train_csv.shape[0]



# Consider Mean of training data and test data

weighted_mean = (num_train * train_csv['Age'].mean() + num_test * test_csv['Age'].mean()) / (num_test + num_train)

# Data Normalization('Age', 'Fare') requires test data

age_max = test_csv['Age'].max() if train_csv['Age'].max() < test_csv['Age'].max() else train_csv['Age'].max()

fare_max = test_csv['Fare'].max() if train_csv['Fare'].max() < test_csv['Fare'].max() else train_csv['Fare'].max()

print(age_max, fare_max)



def Age_prep(x): # 0 ~ 80 -> -1 ~ 1

    return (x/age_max)*2 - 1



def Fare_prep(x): # 0 ~ 512.3 -> -1 ~ 1

    return (x/fare_max)*2 - 1



train_csv['Age'] = train_csv['Age'].apply(Age_prep)

train_csv['Fare'] = train_csv['Fare'].apply(Fare_prep)
## Data Check ##

train_csv.head(5)
# Separate X(data) and Y(Label)

X, y = train_csv[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked',]], train_csv[['Survived']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=9472)
################################

###### 2. Model Training #######

################################



# No reason, Just XGBoost..

xgb_clf = XGBClassifier(n_estimators=20, random_state=5042)

xgb_clf.fit(X_train, y_train, eval_metric="auc", eval_set=[(X_train, y_train),(X_val,y_val)])

xgb_acc_score = classification_report(y_val, xgb_clf.predict(X_val))

print(xgb_acc_score)
################################

##### 3. Predict Test Data #####

################################

test_csv.info()
# Preprocessing of test data is same with training data

test_csv.drop('PassengerId', axis=1, inplace=True)

test_csv.drop('Name', axis=1, inplace=True)

test_csv.drop('Ticket', axis=1, inplace=True)



test_csv['Embarked'].fillna('S',inplace=True)

test_csv.replace({'Embarked': 'S'}, -1, inplace=True)

test_csv.replace({'Embarked': 'C'}, 1, inplace=True)

test_csv.replace({'Embarked': 'Q'}, 0, inplace=True)



test_csv_isnull = test_csv.isnull()

test_csv['Cabin'] = test_csv_isnull['Cabin']

test_csv.replace({'Cabin': True}, 1, inplace=True)

test_csv.replace({'Cabin': False}, -1, inplace=True)



test_csv.replace({'Sex': 'female'}, 1, inplace=True)

test_csv.replace({'Sex': 'male'}, -1, inplace=True)



test_csv['Pclass'] = test_csv['Pclass'].apply(lambda x:-x+2)

test_csv['SibSp'] = test_csv['SibSp'].apply(SibSp_prep)

test_csv['Parch'] = test_csv['Parch'].apply(Parch_prep)



test_csv['Age'].fillna(weighted_mean_age, inplace=True)

test_csv['Age'] = test_csv['Age'].apply(Age_prep)



test_csv['Fare'].fillna(weighted_mean_fare, inplace=True)

test_csv['Fare'] = test_csv['Fare'].apply(Fare_prep)



## Data Check ##

test_csv.head(5)
X_test = test_csv[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]



xgb_clf.predict(X_test)
################################

# 4. Submit Prediction Result ##

################################



# gender_submission.csv is an example of what a submission file should look like

# These predictions assume only female passengers survive

gender_submission_csv = pd.read_csv("../input/titanic-machine-learning-from-disaster/gender_submission.csv")

gender_submission_csv.head(5)
gender_submission_csv['Survived'] = xgb_clf.predict(X_test)

gender_submission_csv.to_csv('pred_titanic.csv', index=False)

gender_submission_csv.head(5)