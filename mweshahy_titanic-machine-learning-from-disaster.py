# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
# Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
# Input data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Cleaning and separation
# Convert Ticket to tkt_code
all_tickets = pd.DataFrame()
all_tickets['Ticket'] = pd.concat([train['Ticket'], test['Ticket']]).reset_index(drop=True)
all_tickets['Ticketa'] = all_tickets['Ticket'].str.extract('(.+) ', expand=False)
#print(all_tickets.Ticketa.isnull().sum())
#print(all_tickets['Ticketa'].unique())
lb_tkt = LabelEncoder()
all_tickets['Ticketa'].fillna('', inplace=True)
all_tickets['tkt_code'] = lb_tkt.fit_transform(all_tickets['Ticketa'])
#all_tickets['tkt_code'] = all_tickets["tkt_code"]/max(all_tickets.tkt_code)
#train['tkt_code'] = np.array(all_tickets['tkt_code'][0:len(train)])
#test['tkt_code'] = np.array(all_tickets['tkt_code'][len(train):len(all_tickets)])

combine = [train, test]
for dataset in combine:
    # Create new columns
    dataset['cbn_pst'] = np.where(dataset.Cabin.notnull(), 1, 0)
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Countess', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Lady', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Dona', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Don', 'Mr')
    dataset['Title'] = dataset['Title'].replace('Major', 'Mr')
    dataset['Title'] = dataset['Title'].replace('Capt', 'Mr')
    dataset['Title'] = dataset['Title'].replace('Jonkheer', 'Mr')
    dataset['Title'] = dataset['Title'].replace('Rev', 'Mr')
    dataset['Title'] = dataset['Title'].replace('Col', 'Mr')
    dataset['Title'] = dataset['Title'].replace('Sir', 'Mr')
    dataset.loc[(dataset['Title']=='Dr') & (dataset['Sex']=='male'), 'Title'] = 'Mr'
    dataset.loc[(dataset['Title']=='Dr') & (dataset['Sex']=='female'), 'Title'] = 'Mrs'
    dataset.drop(['Parch', 'SibSp', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
    # Fill missing
    ## Fill Age with the median age of similar rows according to Pclass, FamilySize and Title
    indices = list(dataset["Age"][dataset["Age"].isnull()].index)
    for i in indices :
        age_med = dataset["Age"].median()
        age_pred = dataset["Age"][((dataset['FamilySize'] == dataset.iloc[i]["FamilySize"]) & (dataset['Title'] == dataset.iloc[i]["Title"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred) :
            dataset['Age'].iloc[i] = age_pred
        else :
            dataset['Age'].iloc[i] = age_med
        
    dataset['Age'] = (dataset['Age']/10).astype(int)
    dataset['Embarked'].fillna(dataset['Embarked'].dropna().mode(), inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
    dataset.loc[dataset['Fare'] > 0, 'Fare'] = round(np.log(dataset['Fare']),2)
    
    # Convert to numericals [Sex, Embarked, Title]
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
    for col in dataset.select_dtypes('object').columns:
        dataset[col] = dataset[col].astype('category')
        dataset[col] = dataset[col].cat.codes
    
# Separation
#train = train.sample(frac=1).reset_index(drop=True)
x_train = train.drop(['Survived','PassengerId'], axis=1)
y_train = train['Survived']
x_test = test.drop(['PassengerId'], axis=1)
import matplotlib.pyplot as plt
for col in train.columns:
    print(pd.crosstab(train[col],train['Survived']))
# Classification
names = ["LogisticRegression", "Nearest Neighbors", "Linear SVM", "SGD",
          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
          "Naive Bayes", "SVC", "voting"]
classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(),
    SVC(kernel="linear", C=0.025),
    SGDClassifier(max_iter=1000),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10),
    MLPClassifier(max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(),
    VotingClassifier(estimators=[('adac', AdaBoostClassifier()),
                                 ('logc', LogisticRegression()),
                                 ('nn', MLPClassifier(max_iter=1000)),
                                 ('svc', SVC(probability=True)),
                                 ('rfc', RandomForestClassifier(n_estimators=10))], voting='soft')
]
classifiers2 = [
    LogisticRegression(),
    KNeighborsClassifier(),
    SVC(kernel="linear", C=0.025),
    SGDClassifier(max_iter=1000),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10),
    MLPClassifier(max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(),
    VotingClassifier(estimators=[('adac', AdaBoostClassifier()),
                                 ('logc', LogisticRegression()),
                                 ('nn', MLPClassifier(max_iter=1000)),
                                 ('svc', SVC(probability=True)),
                                 ('rfc', RandomForestClassifier(n_estimators=10))], voting='soft')
]

best_score = 0
limit = 700
pos = 0
for name, clf in zip(names, classifiers):
    clf.fit(x_train[:limit], y_train[:limit])
    y_pred = clf.predict(x_test)
    score = clf.score(x_train[limit:], y_train[limit:])
    if score > best_score:
        best_model = pos
        best_score = score
        print(pos, name, "becomes the best")
    print(pos, name, score, sum(y_pred)/418)
    pos += 1

classifiers2[10].fit(x_train, y_train)
best_pred = classifiers2[10].predict(x_test)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": best_pred
    })
# Submission
print (submission.groupby("Survived").count().reset_index()["PassengerId"][0])
expected_range = [int(0.9*418*953/1333), int(1.1*418*953/1333)]
print(expected_range)
submission.to_csv('submission.csv', index=False)
if submission.groupby("Survived").count().reset_index()["PassengerId"][0] not in range(expected_range[0], expected_range[1]):
    print(submission.groupby("Survived").count().reset_index()["PassengerId"][0], 'out of expected range', expected_range[0], expected_range[1])
