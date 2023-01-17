# remove warnings

import warnings

warnings.filterwarnings('ignore')

# ---



%matplotlib inline

import pandas as pd

pd.options.display.max_columns = 100

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

import numpy as np



pd.options.display.max_rows = 100
data = pd.read_csv('../input/train.csv')
data
data.shape
data.describe()
def get_combined_data():

    # reading train data

    train = pd.read_csv('../input/train.csv')

    

    # reading test data

    test = pd.read_csv('../input/test.csv')



    # extracting and then removing the targets from the training data 

    targets = train.Survived

    train.drop('Survived', 1, inplace=True)

    



    # merging train data and test data for future feature engineering

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop('index', inplace=True, axis=1)

    

    return combined
combined = get_combined_data()
combined.shape
def get_titles():



    global combined

    

    # we extract the title from each name

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated titles

    Title_Dictionary = {

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

    

    # we map each title

    combined['Title'] = combined.Title.map(Title_Dictionary)
get_titles()
combined.describe()
data.head(891).groupby(['Pclass','Sex']).median()
def guess_missing_age():

    

    global combined

    

    # a function that fills the missing values of the Age variable

    

    def fillAges(row, grouped_median):

        if row['Sex']=='female' and row['Pclass'] == 1:

            if row['Title'] == 'Miss':

                return grouped_median.loc['female', 1, 'Miss']['Age']

            elif row['Title'] == 'Mrs':

                return grouped_median.loc['female', 1, 'Mrs']['Age']

            elif row['Title'] == 'Officer':

                return grouped_median.loc['female', 1, 'Officer']['Age']

            elif row['Title'] == 'Royalty':

                return grouped_median.loc['female', 1, 'Royalty']['Age']



        elif row['Sex']=='female' and row['Pclass'] == 2:

            if row['Title'] == 'Miss':

                return grouped_median.loc['female', 2, 'Miss']['Age']

            elif row['Title'] == 'Mrs':

                return grouped_median.loc['female', 2, 'Mrs']['Age']



        elif row['Sex']=='female' and row['Pclass'] == 3:

            if row['Title'] == 'Miss':

                return grouped_median.loc['female', 3, 'Miss']['Age']

            elif row['Title'] == 'Mrs':

                return grouped_median.loc['female', 3, 'Mrs']['Age']



        elif row['Sex']=='male' and row['Pclass'] == 1:

            if row['Title'] == 'Master':

                return grouped_median.loc['male', 1, 'Master']['Age']

            elif row['Title'] == 'Mr':

                return grouped_median.loc['male', 1, 'Mr']['Age']

            elif row['Title'] == 'Officer':

                return grouped_median.loc['male', 1, 'Officer']['Age']

            elif row['Title'] == 'Royalty':

                return grouped_median.loc['male', 1, 'Royalty']['Age']



        elif row['Sex']=='male' and row['Pclass'] == 2:

            if row['Title'] == 'Master':

                return grouped_median.loc['male', 2, 'Master']['Age']

            elif row['Title'] == 'Mr':

                return grouped_median.loc['male', 2, 'Mr']['Age']

            elif row['Title'] == 'Officer':

                return grouped_median.loc['male', 2, 'Officer']['Age']



        elif row['Sex']=='male' and row['Pclass'] == 3:

            if row['Title'] == 'Master':

                return grouped_median.loc['male', 3, 'Master']['Age']

            elif row['Title'] == 'Mr':

                return grouped_median.loc['male', 3, 'Mr']['Age']

    

    grouped_train = combined.head(891).groupby(['Sex','Pclass','Title'])

    grouped_median_train = grouped_train.median()



    grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','Title'])

    grouped_median_test = grouped_test.median()    

    

    combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) 

                                                      else r['Age'], axis=1)

    

    combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) 

                                                      else r['Age'], axis=1)
guess_missing_age()
def title_to_titleClass():



    global combined

    

    def replace(row):

        if  row['Title'] == 'Master': return 5

        if  row['Title'] == 'Royalty': return 4

        if  row['Title'] == 'Officer': return 3

        if  row['Title'] == 'Master': return 2

        return 1

    

    combined['titleClass'] = combined.apply( lambda r : replace(r), axis=1)
title_to_titleClass()
combined.drop('Title', axis=1, inplace=True)

combined.drop('Embarked', axis=1, inplace=True)

combined.drop('Name', axis=1, inplace=True)
def transform_sex():

    

    global combined

    

    def tonum(row):

        if row['Sex'] == 'female': return 1

        return 0



    combined['female'] = combined.apply( lambda r : tonum(r), axis=1)
transform_sex()
combined.drop('Sex', axis=1, inplace=True)

combined.drop('Ticket', axis=1, inplace=True)

combined.drop('PassengerId', axis=1, inplace=True)

combined.Cabin.fillna('U', inplace=True)
def replace_cabin():

    

    global combined

    

    def to_num(row):

        return ord(row['Cabin'][0]) - ord('A') + 1

    

    combined['hut'] = combined.apply(lambda r: to_num(r), axis=1)

    
replace_cabin()
combined.drop('Cabin', axis=1, inplace=True)
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score
combined['groupsize'] = combined.apply(lambda r: r['SibSp']+r['Parch']+1, axis=1)
def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)
def recover_train_test_target():

    global combined

    

    train0 = pd.read_csv('../input/train.csv')

    

    targets = train0.Survived

    train = combined.head(891)

    test = combined.iloc[891:]

    

    return train, test, targets
train, test, targets = recover_train_test_target()
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

clf = clf.fit(train, targets)
model = SelectFromModel(clf, prefit=True)
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



    grid_search.fit(train, targets)

    model = grid_search

    parameters = grid_search.best_params_



    print('Best score: {}'.format(grid_search.best_score_))

    print('Best parameters: {}'.format(grid_search.best_params_))

else: 

    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 

                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    

    model = RandomForestClassifier(**parameters)

    model.fit(train, targets)
compute_score(model, train, targets, scoring='accuracy')