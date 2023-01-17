# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_columns = 100

import math

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')



from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import SelectFromModel

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score



from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# check for substring in big string

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if string.find(big_string, substring) != -1:

            return substring

    print(big_string)

    return np.nan



"""data = pd.read_csv("../input/train.csv")



data['Age'].fillna(data['Age'].median(), inplace=True)

data.describe()"""



def status(feature):

    print('Processing', feature, ': ok')



def get_combined_data():

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    

    targets = train.Survived

    train.drop('Survived', 1, inplace=True)

    

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop('index', inplace=True, axis=1)

    

    return combined



combined = get_combined_data()



def get_titles():

    global combined

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

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

    combined['Title'] = combined.Title.map(Title_Dictionary)

get_titles()



grouped_train = combined.head(891).groupby(['Sex','Pclass','Title'])

grouped_median_train = grouped_train.median()



grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','Title'])

grouped_median_test = grouped_test.median()



def process_age():

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

    

    combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) 

                                                      else r['Age'], axis=1)

    combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) 

                                                      else r['Age'], axis=1)

    status('age')

process_age()



def process_names():

    global combined

    combined.drop('Name', axis=1, inplace=True)

    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')

    combined = pd.concat([combined, titles_dummies], axis=1)

    combined.drop('Title', axis=1, inplace=True)

    status('names')

process_names()



def process_fares():

    global combined

    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)

    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)

    status('fare')

process_fares()



def process_embarked():

    global combined

    combined.head(891).Embarked.fillna('S', inplace=True)

    combined.iloc[891:].Embarked.fillna('S', inplace=True)

    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')

    combined = pd.concat([combined,embarked_dummies],axis=1)

    combined.drop('Embarked',axis=1,inplace=True)

    status('embarked')

process_embarked()



def process_cabin():

    global combined

    combined.Cabin.fillna('U', inplace=True)

    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')

    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)

    status('cabin')

process_cabin()



def process_sex():

    global combined

    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})

    status('sex')

process_sex()



def process_pclass():

    global combined

    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix='Pclass')

    combined = pd.concat([combined, pclass_dummies], axis=1)

    combined.drop('Pclass', axis=1, inplace=True)

    status('pclass')

process_pclass()



def process_ticket():

    global combined

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

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')

    combined = pd.concat([combined, tickets_dummies], axis=1)

    combined.drop('Ticket', inplace=True, axis=1)

    status('ticket')

process_ticket()



def process_family():

    global combined

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)

    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)

    status('family')



process_family()



combined.drop('PassengerId', inplace=True, axis=1)

combined.head()



def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)

    return np.mean(xval)



def recover_train_test_target():

    train0 = pd.read_csv('../input/train.csv')

    targets = train0.Survived

    train = combined.head(891)

    test = combined.iloc[891:]

    return train, test, targets



train, test, targets = recover_train_test_target()



clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

clf = clf.fit(train, targets)



features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)



model = SelectFromModel(clf, prefit=True)

train_reduced = model.transform(train)

train_reduced.shape



test_reduced = model.transform(test)

test_reduced.shape



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



output = model.predict(test).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv('../input/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

#df_output[['PassengerId','Survived']].to_csv('../input/output.csv',index=False)

# Any results you write to the current directory are saved as output.