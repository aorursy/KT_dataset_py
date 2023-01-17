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
def get_combined_data():

    # reading train/test data

    train = pd.read_csv('../input/train.csv')

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

combined.info()
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
combined.head(10)
#Fill missing Age

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
process_age()

combined.info()
def process_names():

    

    global combined

    # we clean the Name variable

    combined.drop('Name',axis=1,inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')

    combined = pd.concat([combined,titles_dummies],axis=1)

    

    # removing the title variable

    combined.drop('Title',axis=1,inplace=True)
def process_names_no_dummy():

    global combined

    combined.drop('Name', axis=1, inplace=True)

    title_map = {

                "Mr":0,

                "Officer":1,

                "Master":2,

                "Royalty":3,

                "Miss":4,

                "Mrs":5,

    }

    combined['Title'] = combined.Title.map(title_map)
#process_names()

process_names_no_dummy()
#process Fare

def process_fares():

    

    global combined

    # there's one missing fare value - replacing it with the mean.

    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)

    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)
process_fares()

combined.info()
#process Embarked

def process_embarked():

    

    global combined

    # two missing embarked values - filling them with the most frequent one (S)

    combined.head(891).Embarked.fillna('S', inplace=True)

    combined.iloc[891:].Embarked.fillna('S', inplace=True)

    

    

    # dummy encoding 

    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')

    combined = pd.concat([combined,embarked_dummies],axis=1)

    combined.drop('Embarked',axis=1,inplace=True)

    
def process_embarked_no_dummy():

    global combined

    combined.head(891).Embarked.fillna('S', inplace=True)

    combined.iloc[891:].Embarked.fillna('S', inplace=True)



    combined['Embarked'] = combined.Embarked.map({'S':0, 'Q':1, 'C':2})

5#process_embarked()

process_embarked_no_dummy()

#combined.drop('Embarked', axis=1, inplace=True)



combined.info()
#Cabin

def process_cabin():

    

    global combined

    

    # replacing missing cabins with U (for Uknown)

    combined.Cabin.fillna('U', inplace=True)

    

    # mapping each Cabin value with the cabin letter

    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

    

    # dummy encoding ...

    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')

    

    combined = pd.concat([combined,cabin_dummies], axis=1)

    

    combined.drop('Cabin', axis=1, inplace=True)
def process_cabin_no_dummy():

    global combined

    combined.drop('Cabin', axis=1, inplace=True)
#process_cabin()

process_cabin_no_dummy()

combined.info()
#Sex

def process_sex():

    

    global combined

    # mapping string values to numerical one 

    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
process_sex()
def process_pclass():

    

    global combined

    # encoding into 3 categories:

    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    

    # adding dummy variables

    combined = pd.concat([combined,pclass_dummies],axis=1)

    

    # removing "Pclass"

    

    combined.drop('Pclass',axis=1,inplace=True)
def process_pclass_no_dummy():

    global combined

    pass
#process_pclass()

process_pclass_no_dummy()
#Ticket

def process_ticket():

    

    global combined

    

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

    def cleanTicket(ticket):

        return ticket[0]

    



    # Extracting dummy variables from tickets:



    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')

    combined = pd.concat([combined, tickets_dummies], axis=1)

    combined.drop('Ticket', inplace=True, axis=1)
def process_ticket_no_dummy():

    global combined

    combined.drop('Ticket', axis=1, inplace=True)
#process_ticket()

process_ticket_no_dummy()
#Family

def process_family():

    

    global combined

    # introducing a new feature : the size of families (including the passenger)

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    

    # introducing other features based on the family size

    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)

    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)

    combined.drop('Parch', axis=1, inplace=True)

    combined.drop('SibSp', axis=1, inplace=True)

    combined.drop('FamilySize', axis=1, inplace=True)
def process_family2():

    #only create FamilySize

    global combined

    # introducing a new feature : the size of families (including the passenger)

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    combined.drop('Parch', axis=1, inplace=True)

    combined.drop('SibSp', axis=1, inplace=True)
#process_family()

process_family2()
combined.shape
combined.drop('PassengerId', inplace=True, axis=1)
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
def recover_train_test_target():

    global combined

    

    train0 = pd.read_csv('../input/train.csv')

    

    targets = train0.Survived

    train = combined.head(891)

    test = combined.iloc[891:]

    

    return train, test, targets
train, test, targets = recover_train_test_target()

train.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

clf = clf.fit(train, targets)
features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))
model = SelectFromModel(clf, prefit=True)

train_reduced = model.transform(train)

train_reduced.shape
test_reduced = model.transform(test)

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



    grid_search.fit(train, targets)

    model = grid_search

    parameters = grid_search.best_params_



    print('Best score: {}'.format(grid_search.best_score_))

    print('Best parameters: {}'.format(grid_search.best_params_))

else: 

    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 'random_state':2,

                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    

    model = RandomForestClassifier(**parameters)

    model.fit(train, targets)
compute_score(model, train, targets, scoring='accuracy')
#submit

output = model.predict(test).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv('../input/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('ahmed8134_no_dummy5.csv',index=False)
#read train and test data set

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
#plot survival rate by gender

train_df[['Sex', 'Survived']].groupby('Sex').mean()
# we extract the title from each name

train_df['Title'] = train_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

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

train_df['Title'] = train_df.Title.map(Title_Dictionary)
print(train_df[['Title', 'Survived']].groupby('Title').mean())

print(train_df[['Embarked', 'Survived']].groupby('Embarked').mean())