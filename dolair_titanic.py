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
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

train_df = train

test_df = test

data = pd.concat([train.drop("Survived", axis=1),test], axis=0).reset_index(drop=True)

data.head()
import matplotlib.pyplot as plt

data.hist(bins=50, figsize=(20,15))

#plt.show()

data.info()
data['Relatives'] = data['SibSp'] + data['Parch']

data = data.drop(columns=["PassengerId"], axis=1)
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}



data['Cabin'] = data['Cabin'].fillna("U0")

data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

data['Deck'] = data['Deck'].map(deck)

data['Deck'] = data['Deck'].fillna(0)

data['Deck'] = data['Deck'].astype(int)

    

# we can now drop the cabin feature

data = data.drop(['Cabin'], axis=1)
#Filling in the missing Age values

missing_age_value = data[data["Age"].isnull()]

for index, row in missing_age_value.iterrows():

    median = data["Age"][(data["Pclass"] == row["Pclass"]) & (data["Embarked"] == row["Embarked"]) & (data["Relatives"] == row["Relatives"])].median()

    if not np.isnan(median):

        data["Age"][index] = median

    else:

        data["Age"][index] = np.median(data["Age"].dropna())
common_value = 'S'



data['Embarked'] = data['Embarked'].fillna(common_value)
print("Pclass of the data point with missing Fare value:", int(data[data["Fare"].isnull()]["Pclass"]))

median = data[data["Pclass"] == 3]["Fare"].median()

data["Fare"].fillna(median, inplace=True)
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



# extract titles

data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# replace titles with a more common title or as Rare

data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

data['Title'] = data['Title'].replace('Mlle', 'Miss')

data['Title'] = data['Title'].replace('Ms', 'Miss')

data['Title'] = data['Title'].replace('Mme', 'Mrs')

# convert titles into numbers

data['Title'] = data['Title'].map(titles)

# filling NaN with 0, to get safe

data['Title'] = data['Title'].fillna(0)

data = data.drop(['Name'], axis=1)
genders = {"male": 0, "female": 1}



data['Sex'] = data['Sex'].map(genders)
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)
data = pd.get_dummies(data=data, columns=["Embarked"], drop_first=True)
data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0

data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2

data.loc[(data['Fare'] > 31) & (data['Fare'] <= 99), 'Fare']   = 3

data.loc[(data['Fare'] > 99) & (data['Fare'] <= 250), 'Fare']   = 4

data.loc[ data['Fare'] > 250, 'Fare'] = 5

data['Fare'] = data['Fare'].astype(int)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

label = LabelEncoder()

data["Age"] = label.fit_transform(pd.cut(data["Age"].astype(int), 5))

data = pd.get_dummies(data=data, columns=["Age"], drop_first=True)

data.head()
data['Fare_Per_Person'] = data['Fare']/(data['Relatives']+1)

data['Fare_Per_Person'] = data['Fare_Per_Person'].astype(int)

data.drop(["SibSp", "Parch", "Ticket"], inplace=True, axis=1)

data.head(10)
data.info()
#Splitting into train and test again

X_train = data[:891]

Y_train = train["Survived"]

print(X_train.head())

X_test = data[891:]
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,

            random_state=None, verbose=0,

            warm_start=False)

scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
# Random Forest

random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,

            random_state=None, verbose=0,

            warm_start=False)



random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)



acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest)
data_to_submit = pd.DataFrame({

    'PassengerId': test['PassengerId'],

    'Survived': Y_prediction

})

data_to_submit.head()
data_to_submit.to_csv('csv_to_submit.csv', index = False)