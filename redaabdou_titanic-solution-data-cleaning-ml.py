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

        

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

    

# reading test data

test = pd.read_csv('/kaggle/input/titanic/test.csv')

PassengerId=test.PassengerId.values



# extracting and then removing the targets from the training data 

targets = train.Survived

train.drop(['Survived'], 1, inplace=True)

    

# merging train data and test data for future feature engineering

# we'll also remove the PassengerID since this is not an informative feature

combined = train.append(test)

combined.reset_index(inplace=True)

combined.drop(['index', 'PassengerId'], inplace=True, axis=1)
# replacing missing values with the mean value of the train data set

combined.Age.fillna(train.Age.median(), inplace=True)

combined.Fare.fillna(train.Fare.mean(), inplace=True)
# replacing missing values with the most frequent value in the train data set

combined.Embarked.fillna(train.Embarked.mode()[0], inplace=True)



# dummy encoding 

embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')

combined = pd.concat([combined, embarked_dummies], axis=1)



# removing "Embarked"

combined.drop('Embarked', axis=1, inplace=True)
titles = set()

for name in combined['Name']:

    titles.add(name.split(',')[1].split('.')[0].strip())

print(titles)
combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

combined['Title'] = combined['Title'].map({

    "Capt": "Other",

    "Col": "Other",

    "Major": "Other",

    "Jonkheer": "Other",

    "Don": "Other",

    "Dona": "Other",

    "Sir" : "Other",

    "Lady" : "Other",

    "Dr": "Other",

    "Rev": "Other",

    "the Countess":"Other",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master"

})



# encoding in dummy variable

titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')

combined = pd.concat([combined, titles_dummies], axis=1)



# removing the name and title variable

combined.drop(columns=['Name','Title'], axis=1, inplace=True)
# replacing missing cabins with U (for Uknown)

combined.Cabin.fillna('U', inplace=True)

    

# mapping each Cabin value with the cabin letter

combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    

# dummy encoding ...

cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    

combined = pd.concat([combined, cabin_dummies], axis=1)



# removing "Cabin"

combined.drop('Cabin', axis=1, inplace=True)
# mapping string values to numerical one 

combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
# encoding "Pclass" into 3 categories:

pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    

# adding dummy variable

combined = pd.concat([combined, pclass_dummies],axis=1)

    

# removing "Pclass"

combined.drop('Pclass',axis=1,inplace=True)
# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

def cleanTicket(ticket):

    ticket = ticket.replace('.','')

    ticket = ticket.replace('/','')

    ticket = ticket.split()

    ticket = map(lambda t : t.strip(), ticket)

    ticket = list(filter(lambda t : not t.isdigit(), ticket))

    if len(ticket) > 0:

        return ticket[0]

    else: 

        return 'XXX'

    

# Extracting dummy variables from tickets:

combined['Ticket'] = combined['Ticket'].map(cleanTicket)

tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')

combined = pd.concat([combined, tickets_dummies], axis=1)

combined.drop('Ticket', inplace=True, axis=1)
# introducing a new feature : the size of families (including the passenger)

combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    

# introducing other features based on the family size

combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)

combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
targets = targets

train = combined.iloc[:891]

test = combined.iloc[891:]
parameter_grid = {

                 'max_depth' : [4, 6, 8],

                 'n_estimators': [50, 10],

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [2, 3, 10],

                 'min_samples_leaf': [1, 3, 10],

                 'bootstrap': [True, False],

                 }

forest = RandomForestClassifier()

cross_validation = StratifiedKFold(n_splits=5)



grid_search = GridSearchCV(forest,

                               scoring='accuracy',

                               param_grid=parameter_grid,

                               cv=cross_validation,

                               verbose=1

                              )



grid_search.fit(train, targets)

parameters=grid_search.best_params_

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(parameters))
model = RandomForestClassifier(**parameters)

model.fit(train, targets)

output = model.predict(test).astype(int)
#make the submission data frame

submission = {

    'PassengerId': PassengerId,

    'Survived': output

}

solution = pd.DataFrame(submission)

solution.head()
#make the submission file

solution.to_csv('submission.csv',index=False)