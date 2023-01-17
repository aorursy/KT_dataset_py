# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# seaborn , matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# import models from scikit-learn

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
# load titanic data!!

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.head()
train_df.info()

test_df.info()
train_df.describe()
# dropping unneccesary columns 

train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

train_df.head()
test_df = test_df.drop(['Name','Ticket','Cabin'], axis=1)

test_df.head()
# Analyze Pclass 



mean_pclass = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

mean_pclass
sns.barplot(x='Pclass',y='Survived',data=mean_pclass)
train_pclass_dummies = pd.get_dummies(train_df['Pclass'], prefix='Pclass')

train_pclass_dummies.head()
# removing column having low value of correlation

train_pclass_dummies = train_pclass_dummies.drop(['Pclass_3'], axis=1)

train_pclass_dummies.head()
test_pclass_dummies = pd.get_dummies(test_df['Pclass'], prefix='Pclass')

test_pclass_dummies.head()
test_pclass_dummies = test_pclass_dummies.drop(['Pclass_3'], axis=1)

test_pclass_dummies.head()
train_df = pd.concat([train_df, train_pclass_dummies], axis=1)

train_df.head()
test_df = pd.concat([test_df, test_pclass_dummies], axis=1)

test_df.head()
# Analyze sex



mean_sex = train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

mean_sex
sns.barplot(x='Sex',y='Survived',data=mean_sex)
train_sex_dummies = pd.get_dummies(train_df['Sex'], prefix='Sex')

train_sex_dummies.head()
# removing column 



train_sex_dummies = train_sex_dummies.drop(['Sex_male'], axis=1)

train_sex_dummies.head()
test_sex_dummies = pd.get_dummies(test_df['Sex'], prefix='Sex')

test_sex_dummies.head()
test_sex_dummies = test_sex_dummies.drop(['Sex_male'], axis=1)

test_sex_dummies.head()
train_df = pd.concat([train_df, train_sex_dummies], axis=1)

train_df.head()
test_df = pd.concat([test_df, test_sex_dummies], axis=1)

test_df.head()
# Analyze sibSp



train_sib = train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_sib
sns.barplot(x='SibSp',y='Survived',data=train_sib)
# Analyze Parch



train_parch = train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_parch
sns.barplot(x='Parch',y='Survived',data=train_parch)
train_df['Family'] = train_df['Parch'] + train_df['SibSp']

train_df.head()
# Analyze family



train_family = train_df[["Family", "Survived"]].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_family
sns.barplot(x='Family',y='Survived',data=train_family)
test_df['Family'] = test_df['Parch'] + test_df['SibSp']

test_df.head()
# Embarked



train_df[train_df['Embarked'].isnull()==True]
# let's count each category



sns.countplot(x='Embarked', data=train_df)
# fill the two missing values with the most occurred value, which is "S".



train_df['Embarked'] = train_df['Embarked'].fillna('S')
# check in test data



test_df[test_df['Embarked'].isnull()==True]



# NO missing value
# analyze Embarked 



train_embarked = train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_embarked
sns.barplot(x='Embarked', y='Survived',data=train_embarked)
train_embark_dummies = pd.get_dummies(train_df['Embarked'], prefix='Embarked')

train_embark_dummies.head()
train_embark_dummies = train_embark_dummies.drop(['Embarked_S'], axis=1)

train_embark_dummies.head()
test_embark_dummies = pd.get_dummies(test_df['Embarked'], prefix='Embarked')

test_embark_dummies.head()
test_embark_dummies = test_embark_dummies.drop(['Embarked_S'], axis=1)

test_embark_dummies.head()
train_df = pd.concat([train_df, train_embark_dummies], axis=1)

train_df.head()
test_df = pd.concat([test_df, test_embark_dummies], axis=1)

test_df.head()
# check missing values on Fair



train_df[train_df['Fare'].isnull()==True]



# No missing value
# check for test data



test_df[test_df['Fare'].isnull()==True]
# one missing value let's fill with median 

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# check again

test_df[test_df['Fare'].isnull()==True]
# let's convert float values to int



train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare']    = test_df['Fare'].astype(int)

train_df.head()
# visualize fair range 

train_df['Fare'].plot(kind='hist', figsize=(15,5),bins=100)
#check for missing value

train_df[train_df['Age'].isnull()==True]
# visualize age bands



train_df['Age'].dropna().astype(int).hist(bins=70)
# Fill missing value with median



train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

train_df[train_df['Age'].isnull()==True]
# Filling missing value in test data



test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

test_df[test_df['Age'].isnull()==True]
# converting float to int 



train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)

train_df.head()
X_Train = train_df.drop(['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'], axis=1).copy()

Y_Train = train_df['Survived']

X_Test = test_df.drop(['PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'], axis=1).copy()
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_Train, Y_Train)



Y_pred = logreg.predict(X_Test)



logreg.score(X_Train, Y_Train)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_Train, Y_Train)



Y_pred = random_forest.predict(X_Test)



random_forest.score(X_Train, Y_Train)
# making submission



submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": Y_pred

})

submission.to_csv('titanic.csv', index=False)