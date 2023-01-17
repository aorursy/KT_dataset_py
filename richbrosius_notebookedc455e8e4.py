# remove warnings

import warnings

warnings.filterwarnings('ignore')

# ---



%matplotlib inline

import pandas as pd

pd.options.display.max_columns = 100

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

import numpy as np

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier





pd.options.display.max_rows = 100
data_train = pd.read_csv('../input/train.csv')
data_train.head()
data_train.describe()
data_train['Age'].fillna(data_train['Age'].median(), inplace=True)
data_train.describe()
survived_sex = data_train[data_train['Survived']==1]['Sex'].value_counts()

dead_sex = data_train[data_train['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(13,8))
figure = plt.figure(figsize=(13,8))

plt.hist([data_train[data_train['Survived']==1]['Fare'],data_train[data_train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
plt.figure(figsize=(13,8))

ax = plt.subplot()

ax.scatter(data_train[data_train['Survived']==1]['Age'],data_train[data_train['Survived']==1]['Fare'],c='green',s=40)

ax.scatter(data_train[data_train['Survived']==0]['Age'],data_train[data_train['Survived']==0]['Fare'],c='red',s=40)

ax.set_xlabel('Age')

ax.set_ylabel('Fare')

ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
ax = plt.subplot()

ax.set_ylabel('Average fare')

data_train.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(13,8), ax = ax)
survived_embark = data_train[data_train['Survived']==1]['Embarked'].value_counts()

dead_embark = data_train[data_train['Survived']==0]['Embarked'].value_counts()

df = pd.DataFrame([survived_embark,dead_embark])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(13,8))
# Store our features in a list

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
data_train = pd.read_csv('../input/train.csv')



#Clean Data

# 1) Replace all missing values with the median for that feature column

data_train["Age"] = data_train["Age"].fillna(data_train["Age"].median())



# 2) Convert male and female labels from string to int: male = 0, female = 1

data_train.loc[data_train["Sex"] == "male", "Sex"] = 0

data_train.loc[data_train["Sex"] == "female", "Sex"] = 1

    

# 3) Convert departure points from char to int: S = 0, C = 1, Q = 2

print(data_train["Embarked"].unique())

data_train["Embarked"] = data_train["Embarked"].fillna("S")

data_train.loc[data_train["Embarked"] == "S", "Embarked"] = 0

data_train.loc[data_train["Embarked"] == "C", "Embarked"] = 1

data_train.loc[data_train["Embarked"] == "Q", "Embarked"] = 2
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = cross_validation.KFold(data_train.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, data_train[predictors], data_train["Survived"], cv=kf)
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
kf = cross_validation.KFold(data_train.shape[0], 3, random_state=1)

scores = cross_validation.cross_val_score(alg, data_train[predictors], data_train["Survived"], cv=kf)
print(scores.mean())
# Now Let's Test our mod
#Load the test set

titanic_test = pd.read_csv('../input/test.csv')
#Clean the test set as we did the training set

titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2



# Fix a bad datapoint in the Fares Column which was not in the training set

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
alg.fit(data_train[predictors], data_train["Survived"])

# Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.

predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]



# Must convert outcomes to either died or survived

predictions[predictions > .5] = 1

predictions[predictions <= .5] = 0

predictions = predictions.astype(int)
# Store your results in a submission file

submission_unformatted = pd.DataFrame({"PassengerId": titanic_test["PassengerId"],"Survived":predictions})

submission = submission_unformatted.set_index("PassengerId")

submission.to_csv('../working/simple_submission.csv')
import os

#os.listdir('.')

os.listdir('../working')