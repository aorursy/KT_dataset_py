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
from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""");



data_loc = "../input/train.csv"

data = pd.read_csv(data_loc)

print(data.shape)
data.head()
data.describe()
data['Age'] = data['Age'].fillna(data['Age'].median())

data.describe()
# Let's visualize survival based on the gender.

data['Died'] = 1 - data['Survived']

data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),

                                                          stacked=True, color=['g', 'r']);
 # It looks like male passengers are more likely to succumb. Let's plot the same graph but with ratio instead.

data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True, color=['g', 'r']);
# The Sex variable seems to be a discriminative feature. Women are more likely to survive.

# Let's now correlate the survival with the age variable.

fig = plt.figure(figsize=(25, 7))

sns.violinplot(x='Sex', y='Age', hue='Survived', data=data, split=True, palette={0: "r", 1: "g"});
# As we saw in the chart above and validate by the following:

# Women survive more than men, as depicted by the larger female green histogram

# Now, we see that: The age conditions the survival for male passengers:

# Younger male tend to survive A large number of passengers between 20 and 40 succumb

# The age doesn't seem to have a direct impact on the female survival

# Let's now focus on the Fare ticket of each passenger and see how it could impact the survival.

figure = plt.figure(figsize=(25, 7))

plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], stacked=True, color = ['g','r'],

         bins = 50, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
# Passengers with cheaper ticket fares are more likely to die. Put differently, passengers with more expensive tickets, and 

# therefore a more important social status, seem to be rescued first.  Let's now combine the age, the fare and the survival.

plt.figure(figsize=(25, 7))

ax = plt.subplot()



ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], 

           c='green', s=data[data['Survived'] == 1]['Fare'])

ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], 

           c='red', s=data[data['Survived'] == 0]['Fare'])
# The size of the circles is proportional to the ticket fare.

# On the x-axis, we have the ages and the y-axis, we consider the ticket fare. We can observe different clusters:

# Large green dots between x=20 and x=45: adults with the largest ticket fares

# Small red dots between x=10 and x=45, adults from lower classes on the boat

# Small greed dots between x=0 and x=7: these are the children that were saved

# As a matter of fact, the ticket fare correlates with the class as we see it in the chart below.

ax = plt.subplot()

ax.set_ylabel('Average fare')

data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(25, 7), ax = ax);
# Let's now see how the embarkation site affects the survival.

fig = plt.figure(figsize=(25, 7))

sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=data, split=True, palette={0: "r", 1: "g"});
# In the previous part, we flirted with the data and spotted some interesting correlations.

# In this part, we'll see how to process and transform these variables in such a way the data becomes manageable by a machine learning algorithm.

# We'll also create, or "engineer" additional features that will be useful in building the model.

# We'll see along the way how to process text variables like the passenger names and integrate this information in our model.

# We will break our code in separate functions for more clarity.

# But first, let's define a print function that asserts whether or not a feature has been processed.

def status(feature):

    print ('Processing', feature, ': ok')
# Loading the data¶ : One trick when starting a machine learning problem is to append the training set to the test set together.

# We'll engineer new features using the train set to prevent information leakage and add these variables to the test set.

# Let's load the train and test sets and append them together.

def get_combined_data():

    # reading train data

    train_loc = "../input/train.csv"

    test_loc = "../input/test.csv"

    

    train = pd.read_csv(train_loc)

    

    # reading test data

    test = pd.read_csv(test_loc)



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
# Let's first see what the different titles are in the train set

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





combined['Title'] = combined["Name"].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated title

    # we map each title

combined['Title'] = combined.Title.map(Title_Dictionary)

status('Title')
combined.head()
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

combined.head()
# Let's now process the names.

def process_names():

    global combined

    # we clean the Name variable

    combined.drop('Name', axis=1, inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')

    combined = pd.concat([combined, titles_dummies], axis=1)

    

    # removing the title variable

    combined.drop('Title', axis=1, inplace=True)

    

    status('Names')

    return combined
combined = process_names()

combined.head()
# there is no longer a name feature. new variables (Title_X) appeared. These features are binary.

# For example, If Title_Mr = 1, the corresponding Title is Mr.

# Let's imputed the missing fare value by the average fare computed on the train set

def process_fares():

    global combined

    # there's one missing fare value - replacing it with the mean.

    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)

    status('Fare')

    return combined
# This function simply replaces one missing Fare value by the mean.

combined = process_fares()

combined.head()
# Processing Embarked

def process_embarked():

    global combined

    # two missing embarked values - filling them with the most frequent one in the train  set(S)

    combined.Embarked.fillna('S', inplace=True)

    # dummy encoding 

    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')

    combined = pd.concat([combined, embarked_dummies], axis=1)

    combined.drop('Embarked', axis=1, inplace=True)

    status('Embarked')

    return combined
# This functions replaces the two missing values of Embarked with the most frequent Embarked value.

combined = process_embarked()

combined.head()
# Processing Cabin

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
# We don't have any cabin letter in the test set that is not present in the train set.

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
# Processing Sex

def process_sex():

    global combined

    # mapping string values to numerical one 

    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})

    status('Sex')

    return combined
# This function maps the string values male and female to 1 and 0 respectively.

combined = process_sex()

combined.head()
# Processing Pclass

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
# This function encodes the values of Pclass (1,2,3) using a dummy encoding.

combined = process_pclass()

combined.head()
# Dropping Ticket Column

combined = combined.drop("Ticket", axis=1)

combined.head()
# Processing Family

# This part includes creating new variables based on the size of the family the size is by the way, another variable we create.

# This creation of new variables is done under a realistic assumption: Large families are grouped together, hence they are

# more likely to get rescued than people traveling alone.

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
# This function introduces 4 new features:

# FamilySize : the total number of relatives including the passenger (him/her)self.

# Sigleton : a boolean variable that describes families of size = 1

# SmallFamily : a boolean variable that describes families of 2 <= size <= 4

# LargeFamily : a boolean variable that describes families of 5 < size

combined = process_family()

print (combined.shape)
# Let's start by importing the useful libraries.

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# To evaluate our model we'll be using a 5-fold cross validation with the accuracy since it's the metric that the 

# competition uses in the leaderboard.

def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)
def recover_train_test_target():

    global combined

    

    targets = data['Survived'].values

    train = combined.iloc[:891]

    test = combined.iloc[891:]

    

    return train, test, targets
train, test, targets = recover_train_test_target()
# Feature selection¶

# We've come up to more than 30 features so far. This number is quite large.

# When feature engineering is done, we usually tend to decrease the dimensionality by selecting the "right" number of 

# features that capture the essential.

# In fact, feature selection comes with many benefits:

# It decreases redundancy among the data

# It speeds up the training process

# It reduces overfitting

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

clf.fit(train, targets)

# Let's have a look at the importance of each feature.

features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(25, 25))
model = SelectFromModel(clf, prefit=True)

train_reduced = model.transform(train)

print (train_reduced.shape)
# Let's try different base models

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

aux = pd.read_csv("../input/test.csv")

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('./gridsearch_rf.csv', index=False)