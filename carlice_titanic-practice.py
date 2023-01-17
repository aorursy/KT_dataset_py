import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# import datasets

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head(10)
train_df.info() # Age: 177 missing; Embarked: 2 missing

print('---------------------------------------')

test_df.info() # Age: 86 missing; Embarked: 1 missing
# drop column 'Name', 'PassengerId', 'Ticket' since they won't be useful for data analysis

# drop column 'Cabin' since it has too much missing data

train_df = train_df.drop(['PassengerId', 'Name','Ticket', 'Cabin'], axis = 1)

test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
# Pclass



sns.factorplot('Pclass', 'Survived', data = train_df)



# create dummy valuables for Pclass

train_pclass_dum = pd.get_dummies(train_df['Pclass'])

train_pclass_dum.columns = ['Class1', 'Class2', 'Class3']



test_pclass_dum = pd.get_dummies(test_df['Pclass'])

test_pclass_dum.columns = ['Class1', 'Class2', 'Class3']



train_df = train_df.join(train_pclass_dum)

test_df = test_df.join(test_pclass_dum)



# drop column 'Pclass'

train_df = train_df.drop('Pclass', axis = 1)

test_df = test_df.drop('Pclass', axis = 1)
# sex

# create dummy variables for sex

train_df['Sex'].loc[train_df['Sex'] == 'male'] = 0

train_df['Sex'].loc[train_df['Sex'] == 'female'] = 1



test_df['Sex'].loc[test_df['Sex'] == 'male'] = 0

test_df['Sex'].loc[test_df['Sex'] == 'female'] = 1
# age

# 177 missing values in train dataset, 86 missing values in test dataset

# calculate the mean, sd and the number of missing value for column age

train_mean_age = train_df['Age'].mean()

train_std_age = train_df['Age'].std()

train_number_age = train_df['Age'].isnull().sum()



test_mean_age = test_df['Age'].mean()

test_std_age = test_df['Age'].std()

test_number_age = test_df['Age'].isnull().sum()



# generate random values between (mean - std) and (mean + std)

train_random = np.random.randint(train_mean_age - train_std_age, train_mean_age + train_std_age, size = train_number_age)

test_random = np.random.randint(test_mean_age - test_std_age, test_mean_age + test_std_age, size = test_number_age)



# drop missing values and fill them with random values

train_df['Age'].dropna()

train_df['Age'][np.isnan(train_df['Age'])] = train_random

test_df['Age'].dropna()

test_df['Age'][np.isnan(test_df['Age'])] = test_random



train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)



# get the age range

train_df['Age'].loc[train_df['Age'] <= 15] = 0

train_df['Age'].loc[(train_df['Age'] > 15) & (train_df['Age'] <= 30)] = 1

train_df['Age'].loc[(train_df['Age'] > 30) & (train_df['Age'] <= 45)] = 2

train_df['Age'].loc[(train_df['Age'] > 45) & (train_df['Age'] <= 60)] = 3



test_df['Age'].loc[test_df['Age'] <= 15] = 0

test_df['Age'].loc[(test_df['Age'] > 15) & (test_df['Age'] <= 30)] = 1

test_df['Age'].loc[(test_df['Age'] > 30) & (test_df['Age'] <= 45)] = 2

test_df['Age'].loc[(test_df['Age'] > 45) & (test_df['Age'] <= 60)] = 3
# Family

# combine Sibsp and Parch

train_df['Family']= train_df['SibSp'] + train_df['Parch']

train_df['Family'].loc[train_df['Family'] > 0] = 1 # with family member

train_df['Family'].loc[train_df['Family'] == 0] = 0 # alone



test_df['Family']= test_df['SibSp'] + test_df['Parch']

test_df['Family'].loc[test_df['Family'] > 0] = 1 # with family member

test_df['Family'].loc[test_df['Family'] == 0] = 0 # alone
# fare

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



train_df['Fare'].loc[train_df['Fare'] <= 102.466] = 0

train_df['Fare'].loc[(train_df['Fare'] > 102.466) & (train_df['Fare'] <= 204.932)] = 1

train_df['Fare'].loc[(train_df['Fare'] > 204.932) & (train_df['Fare'] <= 307.398)] = 2

train_df['Fare'].loc[(train_df['Fare'] > 307.398) & (train_df['Fare'] <= 409.863)] = 3

train_df['Fare'].loc[train_df['Fare'] > 409.863] = 4



test_df['Fare'].loc[test_df['Fare'] <= 102.466] = 0

test_df['Fare'].loc[(test_df['Fare'] > 102.466) & (test_df['Fare'] <= 204.932)] = 1

test_df['Fare'].loc[(test_df['Fare'] > 204.932) & (test_df['Fare'] <= 307.398)] = 2

test_df['Fare'].loc[(test_df['Fare'] > 307.398) & (test_df['Fare'] <= 409.863)] = 3

test_df['Fare'].loc[test_df['Fare'] > 409.863] = 4



train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare'] = test_df['Fare'].astype(int)
# embarked

sns.countplot(train_df['Embarked'])



# fill the 2 missing values with 'S' since it is the most occurred value in this column

train_df['Embarked'] = train_df['Embarked'].fillna('S')

train_embarked_dum = pd.get_dummies(train_df['Embarked'])

train_embarked_dum.columns = ['S', 'C', 'Q']

train_df.join(train_embarked_dum)

train_df = train_df.drop('Embarked', axis = 1)



test_embarked_dum = pd.get_dummies(test_df['Embarked'])

test_embarked_dum.columns = ['S', 'C', 'Q']

test_df.join(test_embarked_dum)

test_df = test_df.drop('Embarked', axis = 1)
train_df.head(10)
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier
# prepare training datasets

train_x = train_df.drop('Survived', axis = 1)

train_y = train_df['Survived']



test_x = test_df.drop('PassengerId', axis = 1).copy()
# logistic regression

logreg = LogisticRegression()

logreg.fit(train_x, train_y)

Y_pred = logreg.predict(test_x)

logreg.score(train_x, train_y)
# random forest

rand_forest = RandomForestClassifier()

rand_forest.fit(train_x, train_y)

y_pred3 = rand_forest.predict(test_x)

rand_forest.score(train_x, train_y)
# guassian distribution

gaus = GaussianNB()

gaus.fit(train_x, train_y)

y_pred2 = gaus.predict(test_x)

gaus.score(train_x, train_y)
# gradient boosting

grad_boost = GradientBoostingClassifier()

grad_boost.fit(train_x, train_y)

y_pred4 = grad_boost.predict(test_x)

grad_boost.score(train_x, train_y)
# random forest has the highest score based on the results above

# use random forest to predict

submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred3})

submission.to_csv('submission.csv', index = False)