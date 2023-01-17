# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.svm import SVC, LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from scipy import stats

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



%matplotlib inline
# Load in datasets

train = pd.read_csv('/Users/afsaja/Desktop/dsi3/dsi3_projects/project_3/train.csv')

test = pd.read_csv('/Users/afsaja/Desktop/dsi3/dsi3_projects/project_3/test.csv')
# Split out 'PassengerId' column from test and train datasets since it has no explanatory value

train_id = train[['PassengerId']]

test_id = test[['PassengerId']]



train.drop('PassengerId', axis=1, inplace=True)

test.drop('PassengerId', axis=1, inplace=True)
train.shape, test.shape
train.head()
test.head()
# We concatenate the two datasets in order to ensure consistent in data cleaning across both sets

combined = pd.concat((train, test)).reset_index(drop=True)

combined.info()
#Getting a sense of inter-variable relationships

sns.heatmap(combined.corr(), vmax=.8, square=True, cmap="PiYG", annot=True, fmt='.1f');
# Removing the survived column from the dataset and placing it in y_train

y_train = train[['Survived']]

combined.drop('Survived', axis=1, inplace=True)
combined.describe()
# Let's start with Embark since it has only two missing values

combined['Embarked'].value_counts(normalize=True)
# Interested to see how we can fill these two missing values of 'Embarked'. Did not find anything I can infer directly.

# Ended up using the most common value 'S'

combined[combined['Embarked'].isna()]
combined['Embarked'].fillna('S', inplace=True)
# Filling 'Age' column with appropriate values

combined[['Age']].plot(kind='hist', bins=20);
# I will proceed with filling the NaN values in Age with the columns median value

combined['Age'].fillna(value=combined['Age'].median(), inplace=True)
sns.distplot(combined.Age, bins=20, kde=True);
# Let's fill in missing values for the 'Fare' column

combined.Fare.fillna(value=combined['Fare'].median(), inplace=True)
# Using only the first letter of each Cabin code and filling NaNs with 'Unknown' or 'U'

combined['Cabin'].fillna('U', inplace=True)

combined['Cabin'] = combined['Cabin'].map(lambda x: x[0])
combined.info()
# We will proceed with binning the 'Age' column to appropriate strata

def bin_age(data):

    data['Age'] = data['Age'].astype(int)

    data.loc[ data['Age'] <= 11, 'Age'] = 0

    data.loc[(data['Age'] > 11) & (data['Age'] <= 18), 'Age'] = 1

    data.loc[(data['Age'] > 18) & (data['Age'] <= 22), 'Age'] = 2

    data.loc[(data['Age'] > 22) & (data['Age'] <= 27), 'Age'] = 3

    data.loc[(data['Age'] > 27) & (data['Age'] <= 32), 'Age'] = 4

    data.loc[(data['Age'] > 32) & (data['Age'] <= 40), 'Age'] = 5

    data.loc[(data['Age'] > 40) & (data['Age'] <= 66), 'Age'] = 6

    data.loc[ data['Age'] > 66, 'Age'] = 6
bin_age(combined)
# Extracting titles of individuals from 'Name' column

combined['Title'] = combined['Name'].str.extract(' ([A-za-z]+)\.', expand=False)
combined['Title'].value_counts()
# Grouping categories of 'Title' column into fewer groups

def replace_titles(x):

    title = x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:

        return 'Mr'

    elif title in ['Countess', 'Mme', 'Dona']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms', 'Lady']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

combined['Title'] = combined.apply(replace_titles, axis=1)
combined['Title'].value_counts()
# Discovered some 'male' rows where the title was 'Mrs', force fixed them into 'Mr'

combined['Title'][(combined['Sex'] == 'male') & (combined['Title'] == 'Mrs')] = 'Mr'
combined['Family_size'] = combined['Parch'] + combined['SibSp']

sns.distplot(combined['Family_size']);
combined.drop(labels=['Parch', 'SibSp', 'Name', 'Ticket'], axis=1, inplace=True)
combined.info()
# Converting 'Age' and 'Pclass' to object type

combined.Age = combined.Age.astype('category')

combined.Pclass = combined.Pclass.astype('category')

combined.Embarked = combined.Embarked.astype('category')

combined.Sex = combined.Sex.astype('category')

combined.Title = combined.Title.astype('category')

combined.Cabin = combined.Cabin.astype('category')

combined.Family_size = combined.Family_size.astype('category')

combined.info()
data = pd.get_dummies(combined, drop_first=True)

data.shape
len(y_train), len(combined['Cabin'][:len(train)])
sns.barplot(x = combined['Cabin'][:len(train)], y = y_train)
# Splitting data back to train and test

X_train = data.iloc[:len(train),:]

X_test = data.iloc[len(train):,:]

y_train = train[['Survived']]

X_test.shape, X_train.shape, y_train.shape
# Baseline of dataset

baseline = y_train.sum() / len(y_train)

baseline
knn = KNeighborsClassifier()
params_knn = {'n_neighbors':np.arange(1, 21, 1),

            'weights':['uniform', 'distance']

             }



knn_gs = GridSearchCV(knn, params_knn, n_jobs=-1, cv=10, verbose=1)



knn_gs.fit(X_train, y_train)

y_pred_knn = knn_gs.predict(X_test)
knn_gs.best_params_
knn_submission = pd.DataFrame({'PassengerId': test_id.PassengerId, 'Survived': y_pred_knn})

knn_submission.to_csv('knn_submission.csv', index=False)
logreg = LogisticRegression()
params_logreg = {'penalty':['l1','l2'],

            'solver':['liblinear', 'saga']

             }



logreg_gs = GridSearchCV(logreg, params_logreg, n_jobs=-1, cv=10, verbose=1)



logreg_gs.fit(X_train, y_train)

y_pred_logreg = logreg_gs.predict(X_test)
logreg_submission = pd.DataFrame({'PassengerId': test_id.PassengerId, 'Survived': y_pred_logreg})

logreg_submission.to_csv('logreg_2_submission.csv', index=False)
logreg_gs.best_params_
dt = DecisionTreeClassifier()
params_dt = {'criterion':['gini','entropy'],

            'max_depth':np.arange(1,10,1),

            'max_features': [None, 'log2', 'sqrt'],

            'min_samples_split': range(5,30),

            'max_leaf_nodes': [None],

            'min_samples_leaf': range(1,10)

             }



dt_gs = GridSearchCV(dt, params_dt, n_jobs=-1, cv=10, verbose=1)



dt_gs.fit(X_train, y_train)

y_pred_dt = dt_gs.predict(X_test)
dt_submission = pd.DataFrame({'PassengerId': test_id.PassengerId, 'Survived': y_pred_dt})

dt_submission.to_csv('dt_submission.csv', index=False)
dt_gs.best_params_
forest = RandomForestClassifier()
params_forest = {

            'max_depth':np.arange(1,10,1),

            'max_features': [None, 'log2', 'sqrt'],

            'min_samples_split': range(5,30),

            'max_leaf_nodes': [None],

            'min_samples_leaf': range(1,10),

            'n_estimators': [10, 100, 1000]    

             }



forest_gs = GridSearchCV(forest, params_forest, n_jobs=-1, cv=5, verbose=1)
forest_gs.fit(X_train, y_train)

y_pred_forest = forest_gs.predict(X_test)
forest_submission = pd.DataFrame({'PassengerId': test_id.PassengerId, 'Survived': y_pred_forest})

forest_submission.to_csv('forest_submission.csv', index=False)
forest_gs.best_params_
# Plotting feature importance

fi = pd.Series(forest.feature_importances_, X.columns).sort_values()

fi.plot(kind='barh');
svc = SVC()
params_svc = {'C':10.**np.arange(-2, 3),

             'kernel':['linear', 'rbf', 'poly', 'sigmoid']

             }



svc_gs = GridSearchCV(svc, params_svc, n_jobs=-1, cv=10, verbose=1)



svc_gs.fit(X_train, y_train)

y_pred_svc = svc_gs.predict(X_test)
svc_submission = pd.DataFrame({'PassengerId': test_id.PassengerId, 'Survived': y_pred_svc})

svc_submission.to_csv('forest_submission.csv', index=False)
forest_gs.best_params_
ada = AdaBoostClassifier()
params_ada = {'n_estimators': [10, 50, 100, 150], 'learning_rate': np.arange(0.1,1,0.1)}

ada_gs = GridSearchCV(ada, params_ada, n_jobs=-1, cv=10, verbose=1)

ada_gs.fit(X_train, y_train)

y_pred_ada = ada_gs.predict(X_test)
ada_submission = pd.DataFrame({'PassengerId': test_id.PassengerId, 'Survived': y_pred_ada})

ada_submission.to_csv('ada_submission.csv', index=False)
ada_gs.best_params_