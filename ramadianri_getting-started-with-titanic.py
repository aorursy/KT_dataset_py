# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.sample(5)
train.describe(include="all")
test.describe(include="all")
sns.barplot(x="Sex", y="Survived", data=train);
sns.barplot(x='Pclass', y='Survived', data=train);
sns.barplot(x='SibSp', y='Survived', data=train);
sns.barplot(x='Parch', y='Survived', data=train);
#sort the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train);
#make a new boolean column where its values are 1 (recorded cabin) and 0

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



#draw a bar plot of CabinBool vs. survival

sns.barplot(x="CabinBool", y="Survived", data=train);
#combining the train and the test datasets

train['Source'] = 'train'

test['Source'] = 'test'



combined_data = pd.concat([train, test], ignore_index=True)



combined_data.sample(5)
#drop the cabin feature

combined_data.drop(['Cabin'], axis=1, inplace=True)



#create dummy variables from the CabinBool feature (drop_first=True to avoid dummy variable trap)

combined_data = pd.get_dummies(combined_data, columns=['CabinBool'], drop_first=True)



combined_data.head()
combined_data.drop(['Ticket'], axis=1, inplace=True)
#find the highest frequency value in the Embark feature

embarked_mode = combined_data['Embarked'].mode()[0]

embarked_mode
#fill NaN values in the Embarked feature

combined_data.fillna({'Embarked': embarked_mode}, inplace=True)



#create dummy variable from the Embarked feature

combined_data = pd.get_dummies(combined_data, columns=['Embarked'], drop_first=True)



combined_data.head()
#extract title from the Name feature

combined_data['Title'] = combined_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



#count title by sex

combined_data.groupby(['Title','Sex'])['PassengerId'].count().unstack()
#replace various titles with more common names

combined_data['Title'] = combined_data['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')

combined_data['Title'] = combined_data['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

combined_data['Title'] = combined_data['Title'].replace('Mlle', 'Miss')

combined_data['Title'] = combined_data['Title'].replace('Ms', 'Miss')

combined_data['Title'] = combined_data['Title'].replace('Mme', 'Mrs')



#calculate median age by title (exclude the unknown age)

age_by_title = combined_data[combined_data['AgeGroup'] != 'Unknown'].groupby('Title')[['Age']].median()

age_by_title
#replace unknown in the AgeGroup feature based on mean age by title above

title_to_agegroup_map = {'Master':'Baby', 'Miss':'Student', 'Mr':'Young Adult',\

                         'Mrs':'Adult', 'Rare':'Adult', 'Royal':'Adult'}



mapped_unknown_agegroup = combined_data[combined_data['AgeGroup'] == 'Unknown']['Title'].map(title_to_agegroup_map)

combined_data.loc[combined_data['AgeGroup'] == 'Unknown', ['AgeGroup']] = mapped_unknown_agegroup



#create dummy variables from the Title feature

combined_data = pd.get_dummies(combined_data, columns=['Title'], drop_first=True)



#create dummy variables from the AgeGroup feature

combined_data = pd.get_dummies(combined_data, columns=['AgeGroup'], drop_first=True)



#drop the Age feature because can be represented by the AgeGroup feature

combined_data.drop(['Age'], axis = 1, inplace=True)



combined_data.head()
combined_data.drop(['Name'], axis=1, inplace=True)
combined_data = pd.get_dummies(combined_data, columns=['Sex'], drop_first=True)
combined_data.fillna({'Fare': combined_data['Fare'].median()}, inplace=True)
X_train = combined_data[combined_data['Source'] == 'train'].drop(['PassengerId','Survived','Source'], axis=1)

y_train = combined_data[combined_data['Source'] == 'train']['Survived'].astype(int)



X_train.isnull().any()
X_train.head()
X_test = combined_data[combined_data['Source'] == 'test'].drop(['PassengerId','Survived','Source'], axis=1)



X_test.isnull().any()
X_test.head()
#choose a model

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier()
#search grid for optimal parameters

from sklearn.model_selection import GridSearchCV



param_grid = {"max_depth": [None],

              "random_state": [2,5],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}



grid = GridSearchCV(model, param_grid, cv=7)



grid.fit(X_train, y_train)



grid.best_params_
#use the best estimators

model = grid.best_estimator_



#predict

y_predict = model.predict(X_test)
#sava results to a file

results = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_predict})

results.to_csv('my_submission.csv', index=False)