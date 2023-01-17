# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import os 

import warnings

warnings.filterwarnings('ignore')
%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)



import pandas as pd

pd.options.display.max_columns = 100



from matplotlib import pyplot as plt

import numpy as np



import seaborn as sns



import pylab as plot

params = { 

    'axes.labelsize': "large",

    'xtick.labelsize': 'x-large',

    'legend.fontsize': 20,

    'figure.dpi': 150,

    'figure.figsize': [25, 7]

}

plot.rcParams.update(params)
from collections import Counter
data = pd.read_csv('../input/train.csv')
# Outlier detection 



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(data,2,["Age","SibSp","Parch","Fare"])
data.shape
data.head()
data.describe()
data['Age'] = data['Age'].fillna(data['Age'].median())

data.describe()
data['Died'] = 1 - data['Survived']
data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),

                                                          stacked=True, colors=['g', 'r']);
data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), 

                                                           stacked=True, colors=['g', 'r']);
fig = plt.figure(figsize=(25, 7))

sns.violinplot(x='Sex', y='Age', 

               hue='Survived', data=data, 

               split=True,

               palette={0: "r", 1: "g"}

              );
figure = plt.figure(figsize=(25, 7))

plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], 

         stacked=True, color = ['g','r'],

         bins = 50, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend();
plt.figure(figsize=(25, 7))

ax = plt.subplot()



ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], 

           c='green', s=data[data['Survived'] == 1]['Fare'])

ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], 

           c='red', s=data[data['Survived'] == 0]['Fare']);
ax = plt.subplot()

ax.set_ylabel('Average fare')

data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(25, 7), ax = ax);
fig = plt.figure(figsize=(25, 7))

sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=data, split=True, palette={0: "r", 1: "g"});
def status(feature):

    print ("Processing", feature, ": ok")
def get_combined_data():

    # reading train data

    train = pd.read_csv('../input/train.csv')

    

    # reading test data

    test = pd.read_csv('../input/test.csv')



    # extracting and then removing the targets from the training data 

    targets = train.Survived

    train.drop(['Survived'], 1, inplace=True)

    



    # merging train data and test data for future feature engineering

    # we'll also remove the PassengerID since this is not an informative feature

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)

    

    return combined
combined = get_combined_data()

print (combined.shape)

combined.head()

titles = set()

for name in data['Name']:

    titles.add(name.split(',')[1].split('.')[0].strip())

print (titles)

Title_Dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty"

}



def get_titles():

    # we extract the title from each name

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated title

    # we map each title

    combined['Title'] = combined.Title.map(Title_Dictionary)

    status('Title')

    return combined
combined = get_titles()

combined.head()

combined[combined['Title'].isnull()]

print (combined.iloc[:891].Age.isnull().sum())

print (combined.iloc[891:].Age.isnull().sum())

grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])

grouped_median_train = grouped_train.median()

grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

grouped_median_train.head()

def fill_age(row):

    condition = (

        (grouped_median_train['Sex'] == row['Sex']) & 

        (grouped_median_train['Title'] == row['Title']) & 

        (grouped_median_train['Pclass'] == row['Pclass'])

    ) 

    return grouped_median_train[condition]['Age'].values[0]





def process_age():

    global combined

    # a function that fills the missing values of the Age variable

    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

    status('age')

    return combined
combined = process_age()
def process_names():

    global combined

    # we clean the Name variable

    combined.drop('Name', axis=1, inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')

    combined = pd.concat([combined, titles_dummies], axis=1)

    

    # removing the title variable

    combined.drop('Title', axis=1, inplace=True)

    

    status('names')

    return combined
combined = process_names()

combined.head()

def process_fares():

    global combined

    # there's one missing fare value - replacing it with the mean.

    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)

    status('fare')

    return combined
combined = process_fares()

def process_embarked():

    global combined

    # two missing embarked values - filling them with the most frequent one in the train  set(S)

    combined.Embarked.fillna('S', inplace=True)

    # dummy encoding 

    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')

    combined = pd.concat([combined, embarked_dummies], axis=1)

    combined.drop('Embarked', axis=1, inplace=True)

    status('embarked')

    return combined
combined = process_embarked()

combined.head()

train_cabin, test_cabin = set(), set()



for c in combined.iloc[:891]['Cabin']:

    try:

        train_cabin.add(c[0])

    except:

        train_cabin.add('U')

        

for c in combined.iloc[891:]['Cabin']:

    try:

        test_cabin.add(c[0])

    except:

        test_cabin.add('U')

print (train_cabin)

print (test_cabin)

def process_cabin():

    global combined    

    # replacing missing cabins with U (for Uknown)

    combined.Cabin.fillna('U', inplace=True)

    

    # mapping each Cabin value with the cabin letter

    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    

    # dummy encoding ...

    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    

    combined = pd.concat([combined, cabin_dummies], axis=1)



    combined.drop('Cabin', axis=1, inplace=True)

    status('cabin')

    return combined
combined = process_cabin()

combined.head()

def process_sex():

    global combined

    # mapping string values to numerical one 

    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})

    status('Sex')

    return combined

combined = process_sex()

def process_pclass():

    

    global combined

    # encoding into 3 categories:

    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    

    # adding dummy variable

    combined = pd.concat([combined, pclass_dummies],axis=1)

    

    # removing "Pclass"

    combined.drop('Pclass',axis=1,inplace=True)

    

    status('Pclass')

    return combined

combined = process_pclass()

def cleanTicket(ticket):

    ticket = ticket.replace('.', '')

    ticket = ticket.replace('/', '')

    ticket = ticket.split()

    ticket = map(lambda t : t.strip(), ticket)

    ticket = list(filter(lambda t : not t.isdigit(), ticket))

    if len(ticket) > 0:

        return ticket[0]

    else: 

        return 'XXX'

tickets = set()

for t in combined['Ticket']:

    tickets.add(cleanTicket(t))
print (len(tickets))

def process_ticket():

    

    global combined

    

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

    def cleanTicket(ticket):

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        ticket = map(lambda t : t.strip(), ticket)

        ticket = list(filter(lambda t : not t.isdigit(), ticket))

        if len(ticket) > 0 :

            return ticket[0]

        else: 

            return "XXX"

    



    # Extracting dummy variables from tickets:



    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')

    combined = pd.concat([combined, tickets_dummies], axis=1)

    combined.drop('Ticket', inplace=True, axis=1)



    status('Ticket')

    return combined
combined = process_ticket()

def process_family():

    

    global combined

    # introducing a new feature : the size of families (including the passenger)

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    

    # introducing other features based on the family size

    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    

    status('family')

    return combined
combined = process_family()
print (combined.shape)
combined.head()
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)

def recover_train_test_target():

    global combined

    

    targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values

    train = combined.iloc[:891]

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

features.plot(kind='barh', figsize=(25, 25))

model = SelectFromModel(clf, prefit=True)

train_reduced = model.transform(train)

print (train_reduced.shape)

test_reduced = model.transform(test)

print (test_reduced.shape)

logreg = LogisticRegression()

logreg_cv = LogisticRegressionCV()

rf = RandomForestClassifier()

gboost = GradientBoostingClassifier()



models = [logreg, logreg_cv, rf, gboost]

for model in models:

    print ('Cross-validation of : {0}'.format(model.__class__))

    score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')

    print ('CV score = {0}'.format(score))

    print ('****')

# turn run_gs to True if you want to run the gridsearch again.

run_gs = False



if run_gs:

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

    model = grid_search

    parameters = grid_search.best_params_



    print('Best score: {}'.format(grid_search.best_score_))

    print('Best parameters: {}'.format(grid_search.best_params_))

    

else: 

    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 

                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    

    model = RandomForestClassifier(**parameters)

    model.fit(train, targets)

output = model.predict(test).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv('../input/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('gender_submission.csv', index=False)
trained_models = []

for model in models:

    model.fit(train, targets)

    trained_models.append(model)



predictions = []

for model in trained_models:

    predictions.append(model.predict_proba(test)[:, 1])



predictions_df = pd.DataFrame(predictions).T

predictions_df['out'] = predictions_df.mean(axis=1)

predictions_df['PassengerId'] = aux['PassengerId']

predictions_df['out'] = predictions_df['out'].map(lambda s: 1 if s >= 0.5 else 0)



predictions_df = predictions_df[['PassengerId', 'out']]

predictions_df.columns = ['PassengerId', 'Survived']