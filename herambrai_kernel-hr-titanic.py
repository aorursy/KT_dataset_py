# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import training data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
# Import test data

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
# Checking the data types of data elements

train.dtypes
# Checking the correlation among the data elements

train.corr()
# Checking the null values in the datasets

print(train.isnull().sum())

print(test.isnull().sum())
# Checking the distribution of continuous data

import matplotlib.pyplot as plt

fig = plt.figure()

ax1 = fig.add_subplot(121)

ax1.hist(train['Age'])

ax1.set_title("Age")

ax2 = fig.add_subplot(122)

ax2.hist(train['Fare'])

ax2.set_title("Fare")
# Checking the distribution of categorical data

import seaborn as sns

import sys

sns.countplot(x='Embarked',data=train,palette='Set1')

plt.show()
# Since 'Age' and 'Fare' are not normally distributed, so we have to replace the NaN values with median in train and test data

train['Age'].fillna(train['Age'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)

test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Since the most occured value for 'Embarked' is S, so replacing NaN with "S"

train["Embarked"].fillna("S", inplace=True)
# As SibSp and Parch corresponds to same information of additional people, hence creating categorical variable for traveling alone

train['TravelBuds']=train['SibSp']+train['Parch']

train['TravelAlone']=np.where(train['TravelBuds']>0, 0, 1)

test['TravelBuds']=test['SibSp']+test['Parch']

test['TravelAlone']=np.where(test['TravelBuds']>0, 0, 1)
# Performing binning in Age column

#bins = [0, 20, 40, 60, 80]

#train['Age_bin'] = pd.cut(train['Age'], bins)

#test['Age_bin'] = pd.cut(test['Age'], bins)
# Dropping unnecessary columns in train and test data



# For SibSp, Parch and TravelBuds, new variable is created which is 'TravelAlone'

train.drop('SibSp', axis=1, inplace=True)

test.drop('SibSp', axis=1, inplace=True)

train.drop('Parch', axis=1, inplace=True)

test.drop('Parch', axis=1, inplace=True)

train.drop('TravelBuds', axis=1, inplace=True)

test.drop('TravelBuds', axis=1, inplace=True)

# For Cabin, more than 75% data is missing

train.drop('Cabin', axis = 1, inplace=True)

test.drop('Cabin', axis = 1, inplace=True)

# PassengerID is uniuqe for every row and hence doesn't contribute in model building

train.drop('PassengerId', axis = 1, inplace=True)

test.drop('PassengerId', axis = 1, inplace=True)

# Name and Ticket are not relevant in this scenario

train.drop('Name', axis = 1, inplace=True)

test.drop('Name', axis = 1, inplace=True)

train.drop('Ticket', axis = 1, inplace=True)

test.drop('Ticket', axis = 1, inplace=True)

# For Age, Age_bin is created

#train.drop('Age', axis = 1, inplace=True)

#test.drop('Age', axis = 1, inplace=True)
# Checking if any null value still remains in the dataset

print(train.isnull().sum())

print(test.isnull().sum())
# Checking the modified train and test data

print(train.head())

print(test.head())
# Performing one hot encoding for categorical variables

train_mod = pd.get_dummies(data=train, columns=['Sex', 'Embarked'])

train_mod.drop('Sex_female', axis=1, inplace=True)

test_mod = pd.get_dummies(data=test, columns=['Sex', 'Embarked'])

test_mod.drop('Sex_female', axis=1, inplace=True)
# Checking the modified train and test data

print(train_mod.head())

print(test_mod.head())
# Checking the size of modified train and test data

print(train_mod.shape)

print(test_mod.shape)
# Separating dependent and independent variables

X = train_mod.iloc[:, 1:9].values

Y = train_mod.iloc[:, 0].values
# Test data variables

X_test = test_mod.iloc[:, 0:8].values
# Importing mutiple classifiers for Individual Checks and Ensembling

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier
## Importing multiple classifiers

#model = GaussianNB()

#logreg = LogisticRegression()

#tree1 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=5, min_samples_leaf=20)

#SVC_clf = SVC()
## Using Voting classifier

#voting_clf = VotingClassifier(estimators=[('SVC', SVC_clf), ('DTree', tree1), ('LogReg', logreg)], voting='hard', n_jobs = -1)

#voting_clf.fit(X, Y)

#predicted_VC = voting_clf.predict(X_test)
## Using XGBoost

#model = XGBClassifier()

#model.fit(X, Y)

#predicted_XG = model.predict(X_test)
# Decision tree classifier choosen which gives better results than Naive Bayes, SVC, Logistic, Xgboost and Voting classifiers

tree1 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=5, min_samples_leaf=20)

tree1.fit(X, Y)
#Predict Output of DT

predicted_DT= tree1.predict(X_test) 

predicted_DT
#Saving output results to csv file

predicted_df = pd.DataFrame(predicted_DT)

predicted_df.to_csv('out.csv')