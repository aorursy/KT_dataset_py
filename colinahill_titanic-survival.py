# Import libraries

import numpy as np

import pandas as pd

import re

import seaborn as sns

sns.set()



# Import data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

full_data = pd.concat([train_data, test_data])
full_data.info()
full_data.Name[train_data.Age > 18].head(10)

# Mr and Mrs have periods so let's check children

# full_data.Name[train_data.Age < 18].head(10)

# Master and Miss have a period as well, so assume this holds for all titles, for example Dr., Rev. etc
def find_title(name):

    # search name for title with a period and return the match if it's found

    title = re.search(' ([A-Za-z]+)\.', name)

    if title: return title.group(1)

    else: return ""
titles = full_data.Name.apply(find_title)

titles.unique().tolist()
full_data['Title'] = full_data['Name'].apply(find_title)

full_data['Title'] = full_data['Title'].replace('Mme', "Mrs")

full_data['Title'] = full_data['Title'].replace('Ms', "Miss")

full_data['Title'] = full_data['Title'].replace('Mlle', "Miss")

full_data['Title'] = full_data['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Dona'], 'Other')



full_data.groupby('Title').Age.describe()



full_data.iloc[[61,829]]
sns.boxplot(y='Fare',x='Embarked',hue='Pclass',data=full_data)
train_data['Embarked'].fillna('C', inplace=True)
### This is more advanced - ignore for now

# def find_deck(cabin):

#     if(pd.notnull(cabin)):

#         # extract the deck letter and number of cabins

#         t = re.findall('([A-Z])', cabin)

#         return [t[0], len(t)] # deck letter, number of cabins

#     else:

#         return ['', 0]

def find_deck(cabin):

    if(pd.notnull(cabin)): return re.match('([A-Z])', cabin).group(1)

    else: return 'Unknown'
# find_deck(train_data.Cabin[0])

# find_deck(train_data.Cabin[27])

full_data.Cabin.apply(find_deck).unique().tolist()
class_fare = full_data.groupby('Pclass').Fare.median()
class_age = full_data.groupby('Title').Age.median()
for i, df in enumerate([train_data, test_data]):

    

    # Creade dummy variables for Sex and drop original, as well as an unnecessary column (male or female)

    df = df.join(pd.get_dummies(df['Sex']))

    df.drop(['Sex', 'male'], inplace=True, axis=1)

    

    # Fix titles, replacing equivelent titles, and grouping unusual titles into 'Other'

    df['Title'] = df['Name'].apply(find_title)

    df['Title'] = df['Title'].replace('Mme', "Mrs")

    df['Title'] = df['Title'].replace('Ms', "Miss")

    df['Title'] = df['Title'].replace('Mlle', "Miss")

    df['Title'] = df['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Dona'], 'Other')



    # Estimate missing ages - replace missing age with the median value of the group with the same title

    nullAge = df.Age.isnull()

    for title, age in class_age.iteritems():

        df.loc[(nullAge) & (df['Title'] == title), 'Age'] = age



    # Assign numbers to titles and drop name column

    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5})

    df.drop(['Name', 'Title'], inplace=True, axis=1)

    

    # Create dummmy variables for Embarked and remove redundant column

    df = df.join(pd.get_dummies(df['Embarked'], prefix='Embarked'))

    df.drop(['Embarked', 'Embarked_S'], inplace=True, axis=1)

    

    # Fix cabin information by ignoring the cabin number and replace with deck

    # Use dummy variables for deck

#     df['Deck'] = df['Cabin'].apply(find_deck)

#     decks = ["deck_" + d for d in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Unknown']]

#     df['Deck'] = df['Deck'].map({deck : j+1 for j, deck in enumerate(decks)})

#     # Need to reindex as test data doesn't contain every deck

#     df = df.join(pd.get_dummies(df['Deck']).reindex(columns=decks, fill_value=0)) 

#     df.drop(['Cabin', 'Deck'], inplace=True, axis=1)

    df.drop('Cabin', inplace=True, axis=1)



    # Create new FamilySize feature: Number of Sibling/Spouse aboard + Number of Parent/Child aboard + original person

    # Drop unnecessary columns

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

#     df.drop(['SibSp', 'Parch'], inplace=True, axis=1)



    # Bin ages 

    nbins = 5

    labels=[j+1 for j in range(nbins)]

    df['Age'] = pd.qcut(df['Age'], nbins, labels=labels)



    # Create new feature relating Age and Class = Age * Class

    df['AgeTimesClass'] = df['Age'].values * df['Pclass'].values



    # Fill in missing fares based on median of that class, and Group Fares into bins

    nullFare = df.Fare.isnull()

    for c, fare in class_fare.iteritems():

        df.loc[(nullFare) & (df['Pclass'] == c), 'Fare'] = fare

    nbins = 4

    labels=[j+1 for j in range(nbins)]

    df['Fare'] = pd.qcut(df['Fare'], nbins, labels=labels)



    # Create new feature for FarePerPerson = Fare/FamilySize

    df['FarePerPerson'] = df['Fare'].values / df['FamilySize'].values



    # Create dummy variables for Class

    df = df.join(pd.get_dummies(df['Pclass'], prefix='class'))

    df.drop(['Pclass', 'class_3'], inplace=True, axis=1)



    # Ticket feature provides no useful information, so drop it

    df.drop('Ticket', inplace=True, axis=1)



    if i == 0:

        df.drop('PassengerId', inplace=True, axis=1)

        X_train, y_train = df.loc[:, df.columns != 'Survived'], df.loc[:, df.columns == 'Survived']

    else:

        submission_id = df['PassengerId']

        df.drop('PassengerId', inplace=True, axis=1)

        X_test = df
# X_test.info()

X_train.info()
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier



# Function to fit a series of decision trees with a different number of leaf nodes to optimise fit

def get_accuracy(n_splits, max_leaf_nodes, n_estimators, X_train, y_train):

    model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=1)

    kf = KFold(n_splits=n_splits, random_state=1)

    kf_acc = 0

    for train_index, test_index in kf.split(X_train):

        kf_X_train, kf_X_test = X_train.values[train_index], X_train.values[test_index]

        kf_y_train, kf_y_test = y_train.values[train_index].ravel(), y_train.values[test_index].ravel()

        model.fit(kf_X_train, kf_y_train)

        kf_acc += accuracy_score(kf_y_test, model.predict(kf_X_test))

    return kf_acc / n_splits



candidate_max_leaf_nodes = range(5, 100, 5)

n_splits = 5

n_estimators = 500



all_accuracy = [get_accuracy(n_splits, n, n_estimators, X_train, y_train) for n in candidate_max_leaf_nodes]

sns.lineplot(candidate_max_leaf_nodes, all_accuracy)

ind = all_accuracy.index(max(all_accuracy))

best_tree_size = candidate_max_leaf_nodes[ind]

print("best_tree_size = {:d}".format(best_tree_size))

print("Validation accuracy for Random Forest Model: {:.6f}".format(all_accuracy[ind]))



# Initialize model

rf_model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=best_tree_size, random_state=1)

# Fit data

rf_model.fit(X_train, y_train.values.ravel())
# Make submission file of passenger ids and predictions

y_pred = pd.Series(rf_model.predict(X_test))

submission = pd.concat([submission_id, y_pred], axis=1)

submission = submission.rename(columns={0:'Survived'})

submission.to_csv('submisson.csv', index=False)
importances = rf_model.feature_importances_

sns.barplot(importances, X_train.columns)
# from sklearn.ensemble import GradientBoostingClassifier

# clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1,max_leaf_nodes=50, random_state=1, loss='deviance')

# clf.fit(X_train, y_train.values.ravelå())

# accuracy_score(clf.predict(X_test), val_y)