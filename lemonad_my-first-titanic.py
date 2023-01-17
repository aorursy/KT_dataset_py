# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

input_path = "../input"



# Any results you write to the current directory are saved as output.

train_csv = os.path.join(input_path, "train.csv")

test_csv = os.path.join(input_path, "test.csv")
def plot_survival(df, feature, labels=None):

    survived = df[df['Survived']==1][feature].value_counts()

    dead = df[df['Survived']==0][feature].value_counts()

    plot_df = pd.DataFrame([survived, dead])

    plot_df.index = ['Survived', 'Dead']

    ax = plot_df.plot(kind='bar', stacked=True)

    ax.set_ylabel("# passengers")

    if labels:

        ax.legend(labels);
# Read training and testing data from the included CSVs.

data = pd.read_csv(train_csv)      # Has Survived targets.

test_data = pd.read_csv(test_csv)  # Goal: predict Survived data.



missing_counts = data.isnull().sum()

print("Missing values in training data (%d rows):" % len(data))

print(missing_counts[missing_counts > 0].to_string())



missing_counts = test_data.isnull().sum()

print("Missing values in test data (%d rows):" % len(test_data))

print(missing_counts[missing_counts > 0].to_string())



# From training data:

# 'PassengerId': 1 ... 891

# 'Survived': {0, 1}

# 'Pclass': {1, 2, 3}

# 'Name': {'Dr.', 'Countess.', etc.} {'Mrs.', 'Miss.', 'Ms.', 'Mme.', 'Mlle.'}

# 'Sex': {'male', 'female'} --> {0, 1} [577 males, 314 females]

# 'Age': 0.42 ... 80.0

# 'SibSp': {0, 1, 2, 3, 4, 5, 8} (Number of Siblings/Spouses Aboard)

# 'Parch': [0, 6] (Number of Parents/Children Aboard)

# 'Ticket': E.g. 'SC/Paris 2123', '17463', 'LINE', 'S.O./P.P. 751'

# 'Fare': [0.0, 512.3292]

# 'Cabin': E.g. 'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'T'

# 'Embarked': {'C', 'Q', 'S'}  Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton
# Correlation between numerical features.

f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr,

            mask=np.zeros_like(corr, dtype=np.bool),

            cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True,

            ax=ax)
plot_survival(data, 'Pclass')

data.groupby('Pclass').agg({'Survived': ['mean', 'count']})
pclass_map = {1: '1st', 2: '2nd', 3: '3rd'}

data['Pclass'] = data['Pclass'].map(pclass_map)

test_data['Pclass'] = test_data['Pclass'].map(pclass_map)
# Double check that we do not have any names with multiple titles.

# Titles need to be at least two characters to avoid e.g. the A.

# in Anna A. Andersson being treated as a title.

titles = data['Name'].str.extractall('([^ ]{2,}\.)')

test_titles = test_data['Name'].str.extractall('([^ ]{2,}\.)')



print("Number of names with multiple titles: %d" %

      (sum(titles.reset_index()['match'] > 0) +

       sum(test_titles.reset_index()['match'] > 0)))
# Since there are no duplicate titles, we create a new title feature.

titles = data['Name'].str.extract('([^ ]{2,}\.)')

test_titles = test_data['Name'].str.extract('([^ ]{2,}\.)')

unique_titles = set.union(set(pd.unique(titles[0].str.lower())),

                          set(pd.unique(test_titles[0].str.lower())))

print("Unique titles (train+test):", sorted(unique_titles))



data['Title'] = titles

test_data['Title'] = test_titles

# Check that all names have titles.

print("Number of names without titles: %d (0 is expected)" % (

    data['Title'].isnull().sum() + test_data['Title'].isnull().sum()))



data.groupby('Title').agg({'Survived': ['mean', 'count']})
# Create a new feature, aggregating titles (later one-hot encoded).

titles = ['Mr.', 'Miss.', 'Mrs.']

data['TitleGroup'] = data['Title']

test_data['TitleGroup'] = test_data['Title']

data.loc[~data['TitleGroup'].isin(titles), 'TitleGroup'] = 'Other'

test_data.loc[~test_data['TitleGroup'].isin(titles), 'TitleGroup'] = 'Other'

plot_survival(data, 'TitleGroup',  ['Mr', 'Miss', 'Mrs', 'Other'])
plot_survival(data, 'Sex')
# First we need to take care of all the NaNs. Using Title

# seems like a better choice than Sex and Pclass.

median_age_per_title = data.groupby('Title')['Age'].transform('median')

data['Age'].fillna(median_age_per_title, inplace=True)

median_age_per_title = test_data.groupby('Title')['Age'].transform('median')

test_data['Age'].fillna(median_age_per_title, inplace=True)

# Still one NaN left for Ms in the test_data.

median_age_per_title = test_data.groupby('TitleGroup')['Age'].transform('median')

test_data['Age'].fillna(median_age_per_title, inplace=True)



import seaborn as sns

sns.set()  # Seaborn default for plots.

facet = sns.FacetGrid(data, hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0, data['Age'].max()))

facet.add_legend()

 

plt.show()
# Scale age [0, ~1].

age_max = data['Age'].max()

data['Age'] = data['Age'] / age_max

test_data['Age'] = test_data['Age'] / age_max
data.groupby('Embarked').agg({'Survived': ['mean', 'count']})
# Set NaNs to 'S' as that is the most common port.

data['Embarked'].fillna('S', inplace=True)

test_data['Embarked'].fillna('S', inplace=True)
# Correlation between numerical features.

f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr,

            mask=np.zeros_like(corr, dtype=np.bool),

            cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True,

            ax=ax)
facet = sns.FacetGrid(data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, data['Fare'].max()))

facet.add_legend()

plt.show()
# First replace zero fares by NaN.

data['Fare'].replace(0, np.nan, inplace=True)

test_data['Fare'].replace(0, np.nan, inplace=True)



# Fare has strong correlation to Pclass so replace all NaNs by

# using Pclass medians (Median is 61 vs mean 86 for first

# class passengers in training set, probably skewed by a few

# really high ticket prices)

mean_fare_per_pclass = data.groupby("Pclass")["Fare"].transform("mean")

data['Fare'].fillna(mean_fare_per_pclass, inplace=True)

mean_fare_per_pclass = test_data.groupby("Pclass")["Fare"].transform("mean")

test_data['Fare'].fillna(mean_fare_per_pclass, inplace=True)



# Now let's just scale Fare [0, ~1].

fare_max = data['Fare'].max()

data['Fare'] = data['Fare'] / fare_max

test_data['Fare'] = test_data['Fare'] / fare_max
# Use first character of Cabin only (A, B, C, ...)

data['CabinDeck'] = data['Cabin'].str.slice(0, 1)

test_data['CabinDeck'] = test_data['Cabin'].str.slice(0, 1)



cabins = set.union(set(pd.unique(data['CabinDeck'])),

                   set(pd.unique(test_data['CabinDeck'])))

print("Cabins (train+test):", cabins)



data.fillna('NaN').groupby(['Pclass', 'CabinDeck']).agg({'Survived': ['mean', 'count']})
# Max number of cabins is 4 for both training and test set.

data['NumCabins'] = data['Cabin'].str.split().str.len() / 4

data['NumCabins'].fillna(0, inplace=True)

test_data['NumCabins'] = test_data['Cabin'].str.split().str.len() / 4

test_data['NumCabins'].fillna(0, inplace=True)

data.head(30)
missing_counts = data.isnull().sum()

print("Missing values in training data (%d rows) [Cabin is OK]:" % len(data))

print(missing_counts[missing_counts > 0].to_string())



missing_counts = test_data.isnull().sum()

print("Missing values in test data (%d rows) [Cabin is OK]:" % len(test_data))

print(missing_counts[missing_counts > 0].to_string())
display(data.groupby(['Parch', 'SibSp']).agg({'Survived': ['mean', 'count']}))
# Let's just scale SibSp and Parch to [0, ~1].

parch_max = data['Parch'].max()

data['Parch'] = data['Parch'] / parch_max

test_data['Parch'] = test_data['Parch'] / parch_max

sibsp_max = data['SibSp'].max()

data['SibSp'] = data['SibSp'] / sibsp_max

test_data['SibSp'] = test_data['SibSp'] / sibsp_max

data.head()
# Extract PassengerId from test data for csv output.

test_ids = test_data['PassengerId']



# Extract targets for training.

targets = data['Survived']



# Drop features without further relevance.

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'CabinDeck', 'Title']

data.drop(drop_cols, axis=1, inplace=True)

test_data.drop(drop_cols, axis=1, inplace=True)

data.drop(['Survived'], axis=1, inplace=True)



# One-hot encode categorical features.

categorical_features = ['Pclass', 'Sex', 'Embarked', 'TitleGroup']

dummies = pd.get_dummies(data[categorical_features], drop_first=True)

data = data.drop(categorical_features, axis=1)

data = pd.concat([data, dummies], axis=1)

# Same for test data.

test_dummies = pd.get_dummies(test_data[categorical_features], drop_first=True)

test_data = test_data.drop(categorical_features, axis=1)

test_data = pd.concat([test_data, test_dummies], axis=1)



test_data.head(20)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split



X_train, X_val, y_train, y_val = train_test_split(

    data, targets, test_size=0.2, random_state=0)



# Choose the type of classifier. 

clf = RandomForestClassifier()

# Choose some parameter combinations to try.

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [3, 5, 7, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1, 3, 5, 8]

             }

# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, cv=5)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data.

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(sum(y_pred == y_val) / len(y_val))



print(confusion_matrix(y_val, y_pred))  

print(classification_report(y_val, y_pred))



y_pred = clf.predict(test_data)

d = {'PassengerId': test_ids, 'Survived': y_pred}

df = pd.DataFrame(data=d)

df.to_csv('tree_csv_to_submit.csv', index = False)

df.head()
from sklearn import svm

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split



X_train, X_val, y_train, y_val = train_test_split(

    data, targets, test_size=0.2, random_state=0)

# best

#clf = svm.SVC(C=1.3, gamma=1, degree=3, kernel='poly')



clf = svm.SVC(kernel='rbf')



# Choose some parameter combinations to try

parameters = {

    'C': [0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.5], 

    'gamma': [0.001, 0.01, 0.1, 1]

    }

# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, cv=5)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data.

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(sum(y_pred == y_val) / len(y_val))



print(confusion_matrix(y_val, y_pred))  

print(classification_report(y_val, y_pred))



y_pred = clf.predict(test_data)

d = {'PassengerId': test_ids, 'Survived': y_pred}

df = pd.DataFrame(data=d)

df.to_csv('svm_csv_to_submit.csv', index = False)

df.head()