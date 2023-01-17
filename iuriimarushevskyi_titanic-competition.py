# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score


from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Read data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

all_data = [train_data, test_data]
df_data = train_data.append(test_data)
train_data
# Explore the data where values are missing for some of the attributes.
missing_data = df_data.isnull().sum().sort_values(ascending=False)
missing_data
# Fills the missing data with 0 and check it
for dataset in all_data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
    
print('Missing data in train set: ', train_data["Fare"].isnull().sum())
print('Missing data in  test set: ',test_data["Fare"].isnull().sum())
# Fills with the top value
print(df_data['Embarked'].describe())

for dataset in all_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print('Missing data in train set: ',train_data["Embarked"].isnull().sum())
print('Missing data in  test set: ',test_data["Embarked"].isnull().sum())
# Fills missing values for Age with the random value based on the mean age value in regards to the standard deviation and is_null
for dataset in all_data:
    
    mean = train_data["Age"].mean()
    std = test_data["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_data["Age"].astype(int)

print('Missing data in train set: ', train_data["Age"].isnull().sum())
print('Missing data in  test set: ',test_data["Age"].isnull().sum())

# There is a lot of the missing data in Cabin category and there is no sence to fill it.
# Suppose that the letters in the tickets and cabins indicate the deck. 
# Extract Decks letters from the Cabin and Ticket and fill NaN with U deck.

# Change letter to number
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

for dataset in all_data:
    dataset['Cabin'] = dataset['Cabin'].astype(str)
    dataset['Ticket'] = dataset['Ticket'].astype(str)
    dataset['Deck'] = dataset['Cabin'].str.extract('([A-G]+)', expand=False)
    dataset['Deck'] = dataset['Deck'].fillna(dataset['Ticket'].str.extract('([A-G]+)', expand=False))
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(8)
    dataset['Deck'] = dataset['Deck'].astype(int)

print('Missing data in train set: ', train_data["Cabin"].isnull().sum())
print('Missing data in  test set: ',test_data["Cabin"].isnull().sum())

#Extract family groups as woman and children with the same last name and keep family survival rate for all members of it.
df_data = pd.concat([train_data.set_index('PassengerId'), test_data.set_index('PassengerId')], axis=0, sort=False)

# Get the Title
df_data['Title'] = df_data.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

# Get data about women or children
df_data['WomanChild'] = ((df_data.Title == 'Master') | (df_data.Sex == 'female'))

# Get the last name
df_data['LastName'] = df_data.Name.str.split(',').str[0]

# Find Family groups by Last Name 
family = df_data.groupby(df_data.LastName).Survived

df_data['FamilyTotalCount'] = family.transform(lambda s: s[df_data.WomanChild].fillna(0).count())
df_data['FamilyTotalCount'] = df_data.mask(df_data.WomanChild, df_data.FamilyTotalCount - 1, axis=0)
df_data['FamilySurvivedCount'] = family.transform(lambda s: s[df_data.WomanChild].fillna(0).sum())
df_data['FamilySurvivedCount'] = df_data.mask(df_data.WomanChild, df_data.FamilySurvivedCount - df_data.Survived.fillna(0), axis=0)

df_data['FamilySurvivalRate'] = (df_data.FamilySurvivedCount / df_data.FamilyTotalCount.replace(0, np.nan))
df_data['FamilySurvivalRate'] = df_data['FamilySurvivalRate'].fillna(0)

df_data['FamilySurvivalRate'] = df_data['FamilySurvivalRate'].round(2)
df_data['Alone'] = (df_data.FamilyTotalCount == 0).astype(int)
df_data.reset_index(level=0, inplace=True)

# Family_Survival in df_train and df_test:
test_data['FamilySurvivalRate'] = df_data['FamilySurvivalRate'][891:].values
test_data['Alone'] = df_data['Alone'][891:].values
test_data['Title'] = df_data['Title'][891:].values

train_data['FamilySurvivalRate'] = df_data['FamilySurvivalRate'][:891]
train_data['Alone'] = df_data['Alone'][:891]
train_data['Title'] = df_data['Title'][:891]

# Extract amount of family members
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
    dataset['FamilySize'] = dataset['FamilySize'].astype(int) 
  
# Suppose that people with the diiferent title have the different chanses to survive
for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')   
# Suppose the long name have belongs to aristocrats and their owners more likely survive
for dataset in all_data:
    dataset['NameLong'] = dataset.Name.str.len()
# Convert sex in numbers
gender = {"male": 0, "female": 1}
for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map(gender)
# Convert ports to numbers
ports = {"S": 0, "C": 1, "Q": 2}
for dataset in all_data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
# Convert titles to numbers
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in all_data:
    dataset['Title'] = dataset['Title'].map(titles)
    dataset['Title'] = dataset['Title'].fillna(5) 
# Discretize variable into equal-sized buckets 
pd.qcut(train_data['Age'], 8).value_counts()
# Create categories of Age based on that buckets
for dataset in all_data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 21), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 21) & (dataset['Age'] <= 24), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 24) & (dataset['Age'] <= 28), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 32), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 37), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 37) & (dataset['Age'] <= 45), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 45, 'Age'] = 7
# Discretize variable into equal-sized buckets 
pd.qcut(train_data['Fare'], 6).value_counts()
# Create categories of Fare based on that buckets
for dataset in all_data:
    dataset.loc[dataset['Fare'] <= 7.0, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.0) & (dataset['Fare'] <= 8.0), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 8.0) & (dataset['Fare'] <= 14.0), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 14.0) & (dataset['Fare'] <= 26.0), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 26.0) & (dataset['Fare'] <= 52.0), 'Fare'] = 4
    dataset.loc[ dataset['Fare'] > 52.0, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
# Discretize variable into equal-sized buckets 
pd.qcut(train_data['NameLong'], 8).value_counts()
# Create categories of NameLong based on that buckets
for dataset in all_data:
    dataset['NameLong'] = dataset['NameLong'].astype(int)
    dataset.loc[ dataset['NameLong'] <= 18, 'NameLong'] = 0
    dataset.loc[(dataset['NameLong'] > 18) & (dataset['NameLong'] <= 20), 'NameLong'] = 1
    dataset.loc[(dataset['NameLong'] > 20) & (dataset['NameLong'] <= 23), 'NameLong'] = 2
    dataset.loc[(dataset['NameLong'] > 23) & (dataset['NameLong'] <= 25), 'NameLong'] = 3
    dataset.loc[(dataset['NameLong'] > 25) & (dataset['NameLong'] <= 27), 'NameLong'] = 4
    dataset.loc[(dataset['NameLong'] > 27) & (dataset['NameLong'] <= 30), 'NameLong'] = 5
    dataset.loc[(dataset['NameLong'] > 30) & (dataset['NameLong'] <= 38), 'NameLong'] = 6
    dataset.loc[ dataset['NameLong'] > 38, 'NameLong'] = 7
# Set up the matplotlib figure
with sns.color_palette("Blues_d", n_colors=10):
    
    fig, axes = plt.subplots(4, 3, figsize=(24, 24),)
    
    a = sns.barplot(x="Deck", y="Survived", data=train_data, ci=False, ax=axes[0,0])
    b = sns.barplot(x="FamilySurvivalRate", y="Survived", data=train_data, ci=False, ax=axes[0,1])
    c = sns.barplot(x="Title", y="Survived", data=train_data, ci=False, ax=axes[0,2])
    d = sns.barplot(x="Sex", y="Survived", data=train_data, ci=False, ax=axes[1,0])
    e = sns.barplot(x="Embarked", y="Survived", data=train_data, ci=False, ax=axes[1,1])
    f = sns.barplot(x="Pclass", y="Survived", data=train_data, ci=False, ax=axes[1,2])
    g = sns.barplot(x="Alone", y="Survived", data=train_data, ci=False, ax=axes[2,0])
    h = sns.barplot(x="Age", y="Survived", data=train_data, ci=False, ax=axes[2,1])
    i = sns.barplot(x="Fare", y="Survived", data=train_data, ci=False, ax=axes[2,2])
    j = sns.barplot(x="NameLong", y="Survived", data=train_data, ci=False, ax=axes[3,0])
    k = sns.barplot(x="FamilySize", y="Survived", data=train_data, ci=False, ax=axes[3,1])
  
# Decrease number of values
for dataset in all_data:
    dataset.loc[dataset['FamilySize'] > 6, 'FamilySize'] = 7
# Remove columns
train_data = train_data.drop(['PassengerId', 'Name', 'SibSp','Parch', 'Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Name', 'SibSp','Parch', 'Ticket', 'Cabin'], axis=1) 
   
X = train_data.drop(['Survived'], axis=1).copy()
y = train_data["Survived"]

X_test = test_data.drop(['PassengerId'], axis=1).copy()

# Split the data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)
# Get scores for classifiers
models_cl = [DecisionTreeClassifier(), 
                GaussianNB(), 
                SGDClassifier(), 
                MLPClassifier(max_iter=700), 
                RandomForestClassifier(), 
                Perceptron(), 
                svm.SVC(), 
                KNeighborsClassifier(),
                XGBClassifier(),
                GradientBoostingClassifier(),
                BaggingClassifier(),
                GaussianProcessClassifier(),
                AdaBoostClassifier(),
                ExtraTreesClassifier(),
                LinearDiscriminantAnalysis()]

names = []
score = []
accuracy = []


for name in models_cl:
    
    model = name
    model = model.fit(X_train, y_train)
    
    predictions = model.predict(X_val)
        
    names.append(str(name).split('(')[0])
    score.append(model.score(X, y))
    accuracy.append(accuracy_score(predictions, y_val))
       
    print(str(name).split('(')[0])
    print("Score: ", score[-1])
    print("Accuracy: ", accuracy[-1])
    print(f"Correct: {(y_val == predictions).sum()}")
    print(f"Incorrect: {(y_val != predictions).sum()}")
    print()
       
data ={'Name' : names, 'Score' : score, 'Accuracy' : accuracy}
MLAcompare = pd.DataFrame(data)
MLAcompare = MLAcompare.sort_values(['Accuracy'])
MLAcompare
adb = AdaBoostClassifier()
adb.fit(X, y)
importances_adb = pd.DataFrame({'feature':X.columns,'AdaBoost':np.round(adb.feature_importances_,3)})
importances_adb = importances_adb.sort_values('AdaBoost',ascending=False).set_index('feature')

dtc = DecisionTreeClassifier()
dtc.fit(X, y)
importances_dtc = pd.DataFrame({'feature':X.columns,'DecisionTree':np.round(dtc.feature_importances_,3)})
importances_dtc = importances_dtc.sort_values('DecisionTree',ascending=False).set_index('feature')

importances = pd.concat([importances_dtc, importances_adb], axis=1)
print(importances)
importances.plot(kind='bar')
parameter = {'n_estimators': [10, 50, 100, 300], 
             'max_samples': [.01, .03, .05, .1, .25],
             'random_state': [0, 1]
             }

from sklearn.model_selection import GridSearchCV, cross_val_score

bgc = BaggingClassifier()
bgc_best = GridSearchCV(estimator=bgc, param_grid=parameter, n_jobs=-1)
bgc_best.fit(X, y)
bgc_best.best_params_
parameter = {'n_estimators': [10, 50, 100, 300],
             'learning_rate': [.01, .03, .05, .1, .25],
             'random_state': [0, 1]
            }

from sklearn.model_selection import GridSearchCV, cross_val_score

adb = AdaBoostClassifier()
adb_best = GridSearchCV(estimator=adb, param_grid=parameter, n_jobs=-1)
adb_best.fit(X, y)
adb_best.best_params_
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score

estimators=[('bgc', BaggingClassifier(max_samples=0.25, n_estimators=50, random_state=0)),
            ('adb', AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=0))
           ]

grid = VotingClassifier(estimators = estimators , voting = 'soft')
grid.fit(X_train, y_train)
predictions_val = grid.predict(X_val)
grid.fit(X, y)
predictions = cross_val_predict(grid, X, y, cv=10)

print("Score:", round(grid.score(X, y) * 100, 6))
scores = cross_val_score(grid, X, y, scoring='accuracy', cv=10)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
print("Precision:", precision_score(y, predictions))
print("Recall:",recall_score(y, predictions))
print("F1_score: ",f1_score(y, predictions))
print("Accuracy: ",accuracy_score(predictions_val, y_val))


# Make predictions and save to csv file

predictions = grid.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")