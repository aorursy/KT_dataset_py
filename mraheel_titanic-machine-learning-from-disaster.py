# import required libraries



# pandas for data loading and analysis

#numpy for mathematical operations

#for plotting matplolib and seaborn

#for modeling, evalution and prediction scikit-learn

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load dataset

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df = df_train.append(df_test, ignore_index = True, sort = True)
# head of data

df.head()
# description of data

df.describe()
df_train.Survived.value_counts()
df.Sex.value_counts()
# shape of test set consist of 418 rows and 11 columns, it is goal to predict the survival of people so that's why

# the survival feature is not included in test set

df_test.shape
# shape of training set consist of 891 rows and 12 columns

df_train.shape
df.info()
# we have seen that more people died than survived so let's visualize

plt.figure(figsize = (10,8))

sns.kdeplot(df['Age'][df.Survived == 1], shade = True, color = 'r')

sns.kdeplot(df['Age'][df.Survived == 0], shade = True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Age for Surviving Population and Deceased Population')

plt.show()
def barplot(x, y, title):

    sns.barplot(x = x, y = y)

    plt.title(title)
barplot(df['Sex'], df['Survived'], 'Bar plot for Survival with respect to gender')
barplot(df['Embarked'], df['Survived'], 'Bar plot for Survival rate with respect to Port of Embarkation')
barplot(df['Pclass'], df['Survived'], 'Bar plot for Survival rate with respect to passenger class')
# handling missing values in training set

df_train.Age = df_train.Age.fillna(df.Age.median())

df_train.Cabin = df_train.Cabin.fillna('U')



most_embarked = df.Embarked.value_counts().index[0]

df_train.Embarked = df_train.Embarked.fillna(most_embarked)
# checking null values in training set

df_train.isnull().sum()
#normalizing the name columns into a new title column

df_train['Title'] = df_train.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())



normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}

df_train.Title = df_train.Title.map(normalized_titles)

df_train.Title.value_counts()
df_train.Sex = df_train.Sex.map({'female': 0, 'male': 1})

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1

df_train.Embarked = df_train.Embarked.map({'S': 0, 'C': 1, 'Q': 2})

df_train.Title = df_train.Title.map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Officer': 4, 'Royalty': 5})
# handling missing values in test set

df_test.Age = df_test.Age.fillna(df.Age.median())

df_test.Cabin = df_test.Cabin.fillna('U')

df_test.Fare = df_test.Fare.fillna(df.Fare.median())
df_test['Title'] = df_test.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())



normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}

df_test.Title = df_test.Title.map(normalized_titles)

df_test.Title.value_counts()
df_test.Sex = df_test.Sex.map({'female': 0, 'male': 1})

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

df_test.Embarked = df_test.Embarked.map({'S': 0, 'C': 1, 'Q': 2})

df_test.Title = df_test.Title.map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Officer': 4, 'Royalty': 5})
target = df_train['Survived']

train_data = df_train.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Cabin', 'Ticket'], axis = 1)

train_data.head()
df_test = df_test.drop(['PassengerId', 'Name', 'SibSp', 'Ticket', 'Cabin', 'Parch'], axis = 1)

df_test.head()
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
clf = DecisionTreeClassifier(min_samples_leaf = 20, max_leaf_nodes = 7)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
Rclf = RandomForestClassifier(n_estimators = 100, max_depth = 11)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
clf = GradientBoostingClassifier(n_estimators = 60)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
Rclf.fit(train_data, target)
prediction = Rclf.predict(df_test)
submission = pd.DataFrame({

        "PassengerId": PassengerIDs,

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()