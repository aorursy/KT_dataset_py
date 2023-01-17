import pandas as pd

import numpy as np

import re

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

import seaborn as sns

import matplotlib.pyplot as plt
# Importing dataset from csv files



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
# Checking columns for missing values

# train_data - Null values exist in columns Age, Cabin and Embarked



train_data.isna().sum()
test_data.isna().sum()
print(train_data[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean())
sns.barplot(x = "Pclass", y = "Survived", data = train_data);
print(train_data[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean())
sns.barplot(x = 'Sex', y = 'Survived', data = train_data);
# Feature Engineering with the help of SibSp and Parch

# New feature can be Family Size 



train_data['family_size'] = 0

test_data['family_size'] = 0



train_data['family_size'] = train_data['SibSp'] + train_data['Parch'] + 1

test_data['family_size'] = test_data['SibSp'] + test_data['Parch'] + 1



print(train_data[["family_size","Survived"]].groupby(["family_size"], as_index = False).mean())
train_data.head()
# Creating a new column 'is_alone' to check survival of passengers travelling alone



def check_alone(row): 

    """Check whether a passenger travelled alone (returns 1) or with family (returns 0).

    

    """

    if row['family_size'] == 1:

        return 1

    else:

        return 0

    

train_data['is_alone'] = train_data.apply(check_alone, axis = 1)

test_data['is_alone'] = test_data.apply(check_alone, axis = 1)
print(train_data[["is_alone", "Survived"]].groupby(["is_alone"], as_index = False).mean())
sns.barplot(x = 'is_alone', y = 'Survived', data = train_data);
# TODO: Fill the 2 NaN values



e = train_data[train_data.Embarked.isna()]

e
# Both rows have the same Fare value

# Range of values for 'C' satisfies the given fare values.

# Setting a limit on the range of the y-axis



plt.ylim(0,200)

ax = sns.boxplot(x = train_data['Embarked'], y = train_data['Fare'], palette = 'GnBu')

plt.show()
# Fill NaN values with 'C'

train_data['Embarked'] = train_data["Embarked"].fillna('C')
train_data.iloc[[61,829],:]
print(train_data[['Embarked','Survived']].groupby(['Embarked'], as_index = False).mean())
# NaN values filled with random number between (avg - avg std. deviation) and (avg + avg std. deviation)

# 177



avg = train_data['Age'].mean()

std = train_data['Age'].std()

random = np.random.randint(avg - std, avg + std, size = 177) 

train_data.loc[train_data.Age.isna(), 'Age'] = random    # fill with random age values
# TODO: repeat for test_data



avg = test_data['Age'].mean()

std = test_data['Age'].std()

random = np.random.randint(avg - std, avg + std, size = 86)

test_data.loc[test_data.Age.isna(), 'Age'] = random

test_data.Age.isna().sum()
# pd.cut() - useful for going from a continuous variable to a categorical variable. 

# Used here to convert ages to groups of age ranges.



train_data.Age = train_data['Age'].astype(int)    # convert to int to create bins easily

test_data.Age = test_data['Age'].astype(int)

train_data['category_age'] = pd.cut(train_data['Age'], 5)    # 5 bins

print(train_data[["category_age","Survived"]].groupby(["category_age"], as_index = False).mean())
sns.catplot(x = 'Survived', y = 'Age', data = train_data, kind = 'violin');
train_data['category_fare'] = pd.qcut(train_data['Fare'], 4)

print(train_data[['category_fare', 'Survived']].groupby(["category_fare"], as_index = False).mean())
sns.distplot(train_data['Fare'], color = 'b');
# Replace NaN in test_data Fare column



print(test_data.loc[test_data.Fare.isna(), 'Embarked'])    # Get Embarked value for NaN Fare

mean = train_data.loc[train_data.Embarked == 'S', 'Fare'].mean()    # Calculate mean for 'S' fares

test_data.loc[test_data.Fare.isna(), 'Fare'] = mean
def get_title(row):

    name = row['Name']

    title_search = re.search('([A-Za-z]+)\.', name)    # start with any letter, end with period

    if title_search:

        return title_search.group(1)    # .group() returns the strings that were matches

    return ""



train_data['Title'] = train_data.apply(get_title, axis = 1)

test_data['Title'] = test_data.apply(get_title, axis = 1)

train_data.head()
print(train_data[['Title','Survived']].groupby(['Title'], as_index = False).mean())
# Narrowing the titles down further



train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Other')

train_data['Title'] = train_data['Title'].replace(['Mlle', 'Ms'], 'Miss')

train_data['Title'] = train_data['Title'].replace(['Mme'], 'Mrs')



test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Other')

test_data['Title'] = test_data['Title'].replace(['Mlle', 'Ms'], 'Miss')

test_data['Title'] = test_data['Title'].replace(['Mme'], 'Mrs')
train_data.Title.value_counts()
print(train_data[['Title','Survived']].groupby(['Title'], as_index = False).mean())
# Sex

sex_map = {'female': 0, 'male': 1}

train_data['Sex'] = train_data.Sex.map(sex_map).astype(int)

test_data['Sex'] = test_data.Sex.map(sex_map).astype(int)



# Title

title_map = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4}

train_data["Title"] = train_data.Title.map(title_map).astype(int)

test_data["Title"] = test_data.Title.map(title_map).astype(int)



# Embarked

emb_map = {'C': 0, 'Q': 1, 'S': 2}

train_data["Embarked"] = train_data.Embarked.map(emb_map).astype(int)

test_data["Embarked"] = test_data.Embarked.map(emb_map).astype(int)
train_data.head(5)
# 1. Cast to pandas categorical datatype

# 2. get dummies

# 3. Avoiding dummy variable trap by dropping first column for each category



categories = ['Pclass', 'Sex', 'Embarked', 'is_alone', 'Title']



for category in categories:

    train_data[category] = pd.Categorical(train_data[category])

    test_data[category] = pd.Categorical(test_data[category])

    train_data = pd.concat([train_data, pd.get_dummies(train_data[category], prefix = category, drop_first = True)], axis = 1)

    test_data = pd.concat([test_data, pd.get_dummies(test_data[category], prefix = category, drop_first = True)], axis = 1)



# Dropping unecessary rows

# Remember, the submissions file must contain passenger id



passenger_ids = test_data['PassengerId']

train_data = train_data.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'category_age', 'category_fare', 'Embarked', 'is_alone', 'Title'], axis = 1)

test_data = test_data.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'is_alone', 'Title'], axis = 1)

train_data.head()    
test_data.head()
# Split data into train and test sets



X_train = train_data.drop(['Survived'], axis = 1)

y_train = train_data["Survived"]

X_test = test_data.copy()
# Performing hyperparameter optimisation



rf = RandomForestClassifier()

grid_params = {"criterion": ['gini', 'entropy'],

              "n_estimators": [30, 50, 100, 200, 500],

              "min_samples_leaf": [1, 5, 10],

              "min_samples_split": [2,4,8,10]}

grid_cv = GridSearchCV(estimator = rf, param_grid = grid_params, scoring = 'accuracy', cv = 3, verbose = 3)

grid_cv = grid_cv.fit(X_train, y_train)

print("Best accuracy: ",grid_cv.best_score_)

print("Best params: ", grid_cv.best_params_)
# Fit Random Forest Classifier to train set with best params from GridSearch



classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', min_samples_leaf = 1, min_samples_split = 10, random_state = 0)

classifier.fit(X_train, y_train)
# Predict test set results

y_pred = classifier.predict(X_test)
# Creating the submissions file



submission = pd.DataFrame({

    'PassengerId': passenger_ids,

    'Survived': y_pred

})



submission.to_csv('submissions.csv', index = False)