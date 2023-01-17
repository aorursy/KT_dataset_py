# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

#%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.columns.values
train_df.head()
train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.

# Review Parch distribution using `percentiles=[.75, .8]`

# SibSp distribution `[.68, .69]`

# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`



train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
survived_sex = train_df[train_df['Survived']==1]['Sex'].value_counts()

dead_sex = train_df[train_df['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(15,8))
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
survived_sibsp = train_df[train_df['Survived']==1]['SibSp'].value_counts()

dead_sibsp = train_df[train_df['Survived']==0]['SibSp'].value_counts()

df = pd.DataFrame([survived_sibsp,dead_sibsp])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(15,10))
def get_titles(data_frame):

    # we extract the title from each name

    data_frame['Title'] = data_frame['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated titles

    Title_Dictionary = {

                        "Capt":       "Officer",

                        "Col":        "Officer",

                        "Major":      "Officer",

                        "Jonkheer":   "Royalty",

                        "Don":        "Royalty",

                        "Sir" :       "Royalty",

                        "Dr":         "Royalty",

                        "Rev":        "Royalty",

                        "the Countess":"Royalty",

                        "Dona":       "Royalty",

                        "Mme":        "Mrs",

                        "Mlle":       "Miss",

                        "Ms":         "Miss",

                        "Mr" :        "Mr",

                        "Mrs" :       "Mrs",

                        "Miss" :      "Miss",

                        "Master" :    "Master",

                        "Lady" :      "Royalty"

                        }

    # we map each title

    data_frame['Title'] = data_frame.Title.map(Title_Dictionary)

    return data_frame
train_df = get_titles(train_df)

train_df.head()
def process_age(data_frame, g_median):

    # a function that fills the missing values of the Age variable

    # TODO: Could try to use more features to get better median?

    def fillAges(row, grouped_median):

        return grouped_median.loc[row['Sex'], row['Pclass'], row['Title']]['Age']



    data_frame.head(891).Age = data_frame.head(891).apply(lambda r : fillAges(r, g_median) 

                                                          if np.isnan(r['Age']) else r['Age'],

                                                          axis=1)

    return data_frame
grouped_train = train_df.head(891).groupby(['Sex','Pclass','Title'])

grouped_median_train = grouped_train.median()

train_df = process_age(train_df, grouped_median_train)

train_df.head()
def process_names(data_frame):

    # we clean the Name variable

    data_frame.drop('Name',axis=1,inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(data_frame['Title'],prefix='Title')

    data_frame = pd.concat([data_frame,titles_dummies],axis=1)

    

    # removing the title variable

    data_frame.drop('Title',axis=1,inplace=True)

    return data_frame
train_df = process_names(train_df)

train_df.head()
def process_fares(data_frame):    

    # there's one missing fare value - replacing it with the mean.

    # data_frame.head(891).Fare.fillna(data_frame.head(891).Fare.mean(), inplace=True)

    data_frame.drop('Fare',axis=1,inplace=True)

    return data_frame
train_df = process_fares(train_df)

train_df.head()
def process_embarked(data_frame):

    # two missing embarked values - filling them with the most frequent one (S)

    data_frame.head(891).Embarked.fillna('S', inplace=True)

    

    # dummy encoding 

    embarked_dummies = pd.get_dummies(data_frame['Embarked'],prefix='Embarked')

    data_frame = pd.concat([data_frame,embarked_dummies],axis=1)

    data_frame.drop('Embarked',axis=1,inplace=True)

    return data_frame
train_df = process_embarked(train_df)

train_df.head()
def process_cabin(data_frame):

    # replacing missing cabins with U (for Uknown)

    data_frame.Cabin.fillna('U', inplace=True)

    

    # mapping each Cabin value with the cabin letter

    data_frame['Cabin'] = data_frame['Cabin'].map(lambda c : c[0])

    

    # dummy encoding ...

    cabin_dummies = pd.get_dummies(data_frame['Cabin'], prefix='Cabin')

    data_frame = pd.concat([data_frame,cabin_dummies], axis=1)

    data_frame.drop('Cabin', axis=1, inplace=True)

    return data_frame
train_df = process_cabin(train_df)

train_df.head()
def process_sex(data_frame):

    # mapping string values to numerical one 

    data_frame['Sex'] = data_frame['Sex'].map({'male':1,'female':0})

    return data_frame
train_df = process_sex(train_df)

train_df.head()
def process_pclass(data_frame):    

    # encoding into 3 categories:

    pclass_dummies = pd.get_dummies(data_frame['Pclass'], prefix="Pclass")

    

    # adding dummy variables

    train_df = pd.concat([data_frame,pclass_dummies],axis=1)

    

    # removing "Pclass"

    data_frame.drop('Pclass',axis=1,inplace=True)

    return data_frame
train_df = process_pclass(train_df)

train_df.head()
train_df_tickets = set()

test_df_tickets = set()

def process_ticket(data_frame, training=False):

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

    def cleanTicket(ticket):

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        ticket = map(lambda t : t.strip(), ticket)

        ticket = [t for t in ticket if not t.isdigit()]

        # ticket = filter(lambda t : not t.isdigit(), ticket)

        t_val = None

        if len(ticket) > 0:

            t_val = ticket[0]

        else: 

            t_val = 'XXX'



        if not training:

            test_df_tickets.add(t_val)



        if training or t_val in train_df_tickets:

            train_df_tickets.add(t_val)

            return t_val

        else:

            return "XXX"

        

    

    # Extracting dummy variables from tickets:

    data_frame['Ticket'] = data_frame['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(data_frame['Ticket'], prefix='Ticket')

    data_frame = pd.concat([data_frame, tickets_dummies], axis=1)

    data_frame.drop('Ticket', inplace=True, axis=1)

    return data_frame
train_df = process_ticket(train_df, training=True)

train_df.head()
def process_family(data_frame):

    # introducing a new feature : the size of families (including the passenger)

    data_frame['FamilySize'] = data_frame['Parch'] + data_frame['SibSp'] + 1

    

    # introducing other features based on the family size

    data_frame['Singleton'] = data_frame['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    data_frame['SmallFamily'] = data_frame['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)

    data_frame['LargeFamily'] = data_frame['FamilySize'].map(lambda s: 1 if 5<=s else 0)

    return data_frame
train_df = process_family(train_df)

train_df.head()
targets = train_df.Survived

train_df.drop('Survived', 1, inplace=True)



train_df.reset_index(inplace=True)

train_df.drop('index', inplace=True, axis=1)
test_df = get_titles(test_df)



grouped_test = test_df.head(418).groupby(['Sex','Pclass','Title'])

grouped_median_test = grouped_test.median()

test_df = process_age(test_df, grouped_median_test)



test_df = process_names(test_df)

test_df = process_fares(test_df)

test_df = process_embarked(test_df)

test_df = process_cabin(test_df)

test_df = process_sex(test_df)

test_df = process_pclass(test_df)

test_df = process_ticket(test_df)

test_df = process_family(test_df)
train_df.drop('PassengerId', inplace=True, axis=1)

test_df.drop('PassengerId', inplace=True, axis=1)



# train_df['Cabin_U'] = train_df['Cabin_T']

train_df.drop('Cabin_T', inplace=True, axis=1)



for extra in train_df_tickets - test_df_tickets:

    train_df['Ticket_XXX'] = train_df['Ticket_' + extra]

    train_df.drop('Ticket_' + extra, inplace=True, axis=1)
train_df.describe().columns
test_df.describe().columns
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score
def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=30, max_features='sqrt')

clf = clf.fit(train_df, targets)
features = pd.DataFrame()

features['feature'] = train_df.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))
model = SelectFromModel(clf, prefit=True)

train_reduced = model.transform(train_df)

train_reduced.shape
test_reduced = model.transform(test_df)

test_reduced.shape
# turn run_gs to True if you want to run the gridsearch again.

run_gs = False



if run_gs:

    parameter_grid = {

                 'max_depth' : [4, 6, 8],

                 'n_estimators': [50, 10],

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [1, 3, 10],

                 'min_samples_leaf': [1, 3, 10],

                 'bootstrap': [True, False],

                 }

    forest = RandomForestClassifier()

    cross_validation = StratifiedKFold(targets, n_folds=5)



    grid_search = GridSearchCV(forest,

                               scoring='accuracy',

                               param_grid=parameter_grid,

                               cv=cross_validation)



    grid_search.fit(train_df, targets)

    model = grid_search

    parameters = grid_search.best_params_



    print('Best score: {}'.format(grid_search.best_score_))

    print('Best parameters: {}'.format(grid_search.best_params_))

else: 

    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 

                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    

    model = RandomForestClassifier(**parameters)

    model.fit(train_df, targets)
compute_score(model, train_df, targets, scoring='accuracy')
output = model.predict(test_df).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv('../input/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv(index=False)