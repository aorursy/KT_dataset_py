# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re
from statistics import mode
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# Reading data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

# Storing Passenger Id for submission
Id = test.PassengerId
train.head()
test.head()
train.hist(figsize=(14,14), color='green', bins=20)
plt.show()
fig = plt.figure(figsize=(10,10))

sns.barplot(x="Sex", y="Survived", data=train)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
fig = plt.figure(figsize=(10,10))


#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
fig = plt.figure(figsize=(10,10))


#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train)

#I won't be printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
fig = plt.figure(figsize=(10,10))


sns.barplot(x="Parch", y="Survived", data=train)
plt.show()
# Combining Data

dataset = pd.concat([train, test], sort=False, ignore_index=True)

# Visualizing missing values
dataset.isnull().mean().sort_values(ascending=False)
# Checking correlations with Heatmap

fig, axs = plt.subplots(nrows=1, figsize=(13, 13))
sns.heatmap(dataset.corr(), annot=True, square=True, cmap='YlGnBu', linewidths=2, linecolor='black', annot_kws={'size':12})
# Filling in the missing value in `Fare` with its median
dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)


# Filling in the missing value in 'Embarked' with its mode (Value: 'S')
dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Creating 'Title' column
dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
dataset['Title'].unique().tolist()
# This shows the percentage of occurrences for each title. 'Mr' occurs the most often.

dataset['Title'].value_counts(normalize=True)*100
# Replacing less familiar names with more familiar names
dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer')
dataset['Title'] = dataset['Title'].replace(['Jonkheer', 'Master'], 'Master')
dataset['Title'] = dataset['Title'].replace(['Don', 'Sir', 'the Countess', 'Lady', 'Dona'], 'Royalty')
dataset['Title'] = dataset['Title'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs')
dataset['Title'] = dataset['Title'].replace(['Mlle', 'Miss'], 'Miss')
  

# Imputing missing values with 0
dataset['Title'] = dataset['Title'].fillna(0)

dataset['Title'].value_counts()
# Filling the missing values in Age with its median
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
g  = sns.factorplot(x="Parch",y="Survived",data=dataset, size = 8)
g = g.set_ylabels("Survival Percentage")
g  = sns.factorplot(x="SibSp",y="Survived",data=dataset, size = 8)
g = g.set_ylabels("Survival Percentage")
# Family Size = # of Siblings + # of Parents + You
dataset['FamSize'] = dataset['SibSp'] + dataset['Parch'] + 1
g  = sns.factorplot(x="FamSize",y="Survived",data=dataset, size = 8)
g = g.set_ylabels("Survival Percentage")
def family_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
    
dataset['FamLabel']=dataset['FamSize'].apply(family_label)
dataset.head()
plt.figure(figsize=(8, 8))
sns.barplot(x="FamLabel", y="Survived", data=dataset, palette='Blues_d')
plt.show()
dataset['Cabin'] = dataset['Cabin'].fillna('Unknown')
dataset['Deck']=dataset['Cabin'].str.get(0)
plt.figure(figsize=(8, 8))
sns.barplot(x='Deck', y='Survived', data=dataset, palette='ocean')
plt.show()
dataset.drop(['Name', 'Ticket', 'SibSp', 'Parch', 'FamSize', 'Cabin'], axis=1, inplace=True)
dataset.dtypes
label = LabelEncoder()

for col in ['Sex', 'Embarked', 'Deck', 'Title']:
    dataset[col] = label.fit_transform(dataset[col])
# Splitting dataset into train
train = dataset[:len(train)]

# Splitting dataset into test
test = dataset[len(train):]

# Drop labels 'Survived' because there shouldn't be a Survived column in the test data
test.drop(labels=['Survived'], axis=1, inplace=True)
train.head()
test.head()
train['Survived'] = train['Survived'].astype(int)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

# Setting up variables for modelling
y = train.Survived

X = train.drop('Survived', axis=1)
# Logistic Regression
print("Logistic Regression:", cross_val_score(LogisticRegression(), X, y).mean())

# SVC
print("SVC:", cross_val_score(SVC(), X, y).mean())

# Random Forest
print("Random Forest:", cross_val_score(RandomForestClassifier(), X, y).mean())

# GaussianNB
print("GaussianNB:", cross_val_score(GaussianNB(), X, y).mean())

# Decision Tree
print("Decision Tree:", cross_val_score(DecisionTreeClassifier(), X, y).mean())
select = SelectKBest(k = 'all')
final_model = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')

pipeline = make_pipeline(select, final_model)

cv_result = cross_validate(pipeline, X, y, cv= 10)

print("CV Test Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_result['test_score']), \
                                                     np.std(cv_result['test_score'])))
pipeline.fit(X, y)

final_predictions = pipeline.predict(test)
output = pd.DataFrame({'PassengerId': Id, 'Survived': final_predictions})
output.to_csv('submission.csv', index=False)
output.head()