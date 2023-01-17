import pandas as pd

import numpy as np

import scipy.stats as stats

import re

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                               GradientBoostingClassifier, ExtraTreesClassifier,

                               VotingClassifier)

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



PassengerId = test['PassengerId']



train.head()
full_data = [train, test]



# Some features of my own that I have added in

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;

    

# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
y_train = train['Survived'].ravel()

x_train = train.drop(['Survived'], axis=1).values 

x_test = test.values
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 

              'gamma' : [0.001, 0.01, 0.1, 1, 10, 100],

              'random_state':[2020]}



grid_search = GridSearchCV(SVC(), param_grid, cv=5)

grid_search.fit(x_train, y_train)



print('Best parameters: {}'.format(grid_search.best_params_))

print('Best cross-validation: {}'.format(grid_search.best_score_))
clf = SVC(C=10, gamma=0.01, random_state=2020)



clf.fit(x_train, y_train)

my_prediction = clf.predict(x_test)
my_prediction_df = pd.DataFrame({'PassengerId':PassengerId.values,

                                 'Survived':my_prediction})



#If you want to output CSV for submission, please comment out the code below

#my_prediction_df.to_csv('my_submission.csv', index=False)
y_train = train['Survived'].ravel()

x_train = train.drop(['Survived'], axis=1).values 

x_test = test.values
estimators=[

    ('SV', SVC(C=10, gamma=0.01, random_state=2020)),

    ('RF', RandomForestClassifier(n_estimators=1000, max_features='auto', max_samples=0.7, random_state=2020)),

    ('ET', ExtraTreesClassifier(n_estimators=1000, max_features='auto', max_samples=0.7, random_state=2020)),

    ('AB', AdaBoostClassifier(n_estimators=100, learning_rate=0.7, random_state=2020)),

    ('GB', GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, subsample=1.0, max_depth=4, max_features=None, random_state=2020)),

    ('KN', KNeighborsClassifier(n_neighbors=5)),

    ('LR', LogisticRegression(C=1, solver='sag', random_state=2020))

]



base_models = VotingClassifier(estimators=estimators, voting='hard', weights=[1,1,1,1,1,1,2])
base_models.fit(x_train, y_train)

my_prediction = base_models.predict(x_test)
my_prediction_df = pd.DataFrame({'PassengerId':PassengerId.values,

                                 'Survived':my_prediction})



#If you want to output CSV for submission, please comment out the code below

#my_prediction_df.to_csv('my_submission.csv', index=False)
columns=[]

for i in range(len(estimators)):

    columns.append(estimators[i][0])



base_models.fit(x_train, y_train)

df_base_predictions = pd.DataFrame(base_models.transform(x_test), columns=columns)
print(df_base_predictions.shape)

df_base_predictions.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df_base_predictions.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            vmin=0.0, square=True, cmap=colormap, linecolor='white', annot=True)
y_train = train['Survived'].ravel()

x_train = train.drop(['Survived'], axis=1).values 

x_test = test.values
#preparing base models

estimators=[

    ('SV', SVC(C=10, gamma=0.01, probability=True, random_state=2020)),

    ('RF', RandomForestClassifier(n_estimators=1000, max_features='auto', max_samples=0.7, random_state=2020)),

    ('ET', ExtraTreesClassifier(n_estimators=1000, max_features='auto', max_samples=0.7, random_state=2020)),

    ('AB', AdaBoostClassifier(n_estimators=100, learning_rate=0.7, random_state=2020)),

    ('GB', GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, subsample=1.0, max_depth=4, max_features=None, random_state=2020)),

    ('KN', KNeighborsClassifier(n_neighbors=5)),

    ('LR', LogisticRegression(C=1, solver='sag', random_state=2020))

]
#Specify base models and output the second level training array & test array.

def get_oof(estimators, x_train, y_train, x_test, NSPLITS=5):

    

    base_models = VotingClassifier(estimators=estimators, voting='hard')

    

    ntrain = x_train.shape[0]

    ntest = x_test.shape[0]

    skf = StratifiedKFold(n_splits=NSPLITS, shuffle=True)

    

    oof_train = np.empty((ntrain, len(estimators))) #the second level training array

    oof_test = np.empty((ntest, len(estimators))) #the second level test array

    oof_test_skf = np.empty((NSPLITS, ntest, len(estimators)))

    

    for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):

        

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        

        base_models.fit(x_tr, y_tr)

        

        oof_train[test_index,:] = base_models.transform(x_te)

        oof_test_skf[i,:,:] = base_models.transform(x_test)

    

    #take the mode of the array from the NSPLITS output and use it as the final test data.

    oof_test = stats.mode(oof_test_skf ,axis=0)[0].reshape((ntest, len(estimators)))

    return oof_train, oof_test
base_predictions_train, base_predictions_test = get_oof(estimators, x_train, y_train, x_test, 5)
columns=[]

for i in range(len(estimators)):

    columns.append(estimators[i][0])



df_base_predictions_train = pd.DataFrame(base_predictions_train, columns=columns)

df_base_predictions_test = pd.DataFrame(base_predictions_test, columns=columns)
print(df_base_predictions_train.shape)

df_base_predictions_train.head()
print(df_base_predictions_test.shape)

df_base_predictions_test.head()
y_train = train['Survived']

x_train = train.drop(['Survived'], axis=1).join(df_base_predictions_train).astype('int64')

x_test = test.join(df_base_predictions_test).astype('int64')
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(x_train.corr(),linewidths=0.1,vmax=1.0,

            square=True, cmap=colormap, linecolor='white', annot=True)
param_grid = {'n_estimators':[1000], 

              'max_samples': [0.7, 0.8, 0.9, None],

              'random_state':[2020]}



grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)

grid_search.fit(x_train, y_train)



print('Best parameters: {}'.format(grid_search.best_params_))

print('Best cross-validation: {}'.format(grid_search.best_score_))
stacker = grid_search.best_estimator_

stacker.fit(x_train, y_train)



my_prediction = stacker.predict(x_test)
my_prediction_df = pd.DataFrame({'PassengerId':PassengerId.values,

                                 'Survived':my_prediction})



#If you want to output CSV for submission, please comment out the code below

#my_prediction_df.to_csv('my_submission.csv', index=False)