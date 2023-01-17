import numpy as np

import pandas as pd

import re

import lightgbm as lgb

import seaborn as sns

import matplotlib.pyplot as plt

import os

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont



# Loading the data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



# Store our test passenger IDs for easy access

PassengerId = test['PassengerId']



# Showing overview of the train dataset

train.head()
test.head()
def get_best_score(model):

    

    print(model.best_score_)

    print(model.best_params_)

    print(model.best_estimator_)

    

    return model.best_score_
# Copy original dataset in case we need it later when digging into interesting features

# WARNING: Beware of actually copying the dataframe instead of just referencing it

# "original_train = train" will create a reference to the train variable (changes in 'train' will apply to 'original_train')

original_train = train.copy() # Using 'copy()' allows to clone the dataset, creating a different object with the same values



# Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings

full_data = [train, test]



# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



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

# Remove all NULLS in the Fare column

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())



# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



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

    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



# Remove all NULLS in the Age column

for dataset in full_data:

    for title in dataset.Title.unique():

        dataset.loc[(dataset.Age.isnull())&(dataset.Title==title),'Age'] = train.Age[train.Title==title].mean()
# Feature selection: remove variables no longer containing relevant information

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head()
test.head()
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);
X_train = train.iloc[:, 1:]

y_train = train.iloc[:, 0]

X_test = test



X_train.head()
model_default = lgb.LGBMClassifier()



cv_scores = cross_val_score(model_default, X_train, y_train, cv=5, scoring='accuracy')



print('Average accuracy: ', cv_scores.mean())



model_default.fit(X_train, y_train);
from sklearn.model_selection import GridSearchCV



model = lgb.LGBMClassifier()



# Set list of values for each parameter

param_grid = {

    'learning_rate': 10**np.linspace(-2, 0, 4), # from 0.01 to 1

    'n_estimators': [100],

    'num_leaves': np.linspace(4, 20, 4).astype(int),

    'reg_alpha': np.linspace(0, 1, 4),

}



model_grid = GridSearchCV(model, param_grid, cv=5)

model_grid.fit(X_train, y_train)



get_best_score(model_grid);
from sklearn.model_selection import RandomizedSearchCV

import scipy.stats as sps



model = lgb.LGBMClassifier()



# Set random distribution for each parameter

param_grid = {

    'learning_rate': sps.loguniform(0.01, 1),

    'n_estimators': sps.randint(10, 200),

    'num_leaves': sps.randint(2, 50),

    'reg_alpha': sps.uniform(0, 1),

    'reg_lambda': sps.uniform(0, 10),

    'min_split_gain': sps.uniform(0, 1),

}



model_rand = RandomizedSearchCV(model, param_grid, cv=5, n_iter=100)

model_rand.fit(X_train, y_train)



get_best_score(model_rand);
from sklearn.base import clone

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization



class BayesianSearchCV:

    '''

    Bayesian Search with cross validation score.

    

    Arguments:

    

    base_estimator: sklearn-like model

    param_bounds: dict

        hyperparameter upper and lower bounds

        example: {

            'param1': [0, 10],

            'param2': [-1, 2],

        }

    scoring: string or callable

        scoring argument for cross_val_score

    cv: int

        number of folds

    n_iter: int

        number of bayesian optimization iterations

    init_points: int

        number of random iterations before bayesian optimization

    random_state: int

        random_state for bayesian optimization

    int_parameters: list

        list of parameters which are required to be of integer type

        example: ['param1', 'param3']

    '''

    

    def __init__(

        self,

        base_estimator,

        param_bounds,

        scoring,

        cv=5,

        n_iter=50,

        init_points=10,

        random_state=1,

        int_parameters=[],

    ):

        self.base_estimator = base_estimator

        self.param_bounds = param_bounds

        self.cv = cv

        self.n_iter = n_iter

        self.init_points = init_points

        self.scoring = scoring

        self.random_state = random_state

        self.int_parameters = int_parameters

    

    def objective(self, **params):

        '''

        We will aim to maximize this function

        '''

        # Turn some parameters into ints

        for key in self.int_parameters:

            if key in params:

                params[key] = int(params[key])

        # Set hyperparameters

        self.base_estimator.set_params(**params)

        # Calculate the cross validation score

        cv_scores = cross_val_score(

            self.base_estimator,

            self.X_data,

            self.y_data,

            cv=self.cv,

            scoring=self.scoring)

        score = cv_scores.mean()

        return score

    

    def fit(self, X, y):

        self.X_data = X

        self.y_data = y

        

        # Create the optimizer

        self.optimizer = BayesianOptimization(

            f=self.objective,

            pbounds=self.param_bounds,

            random_state=self.random_state,

        )

        

        # The optimization itself goes here:

        self.optimizer.maximize(

            init_points=self.init_points,

            n_iter=self.n_iter,

        )

        

        del self.X_data

        del self.y_data

        

        # Save best score and best model

        self.best_score_ = self.optimizer.max['target']

        self.best_params_ = self.optimizer.max['params']

        for key in self.int_parameters:

            if key in self.best_params_:

                self.best_params_[key] = int(self.best_params_[key])

        

        self.best_estimator_ = clone(self.base_estimator)

        self.best_estimator_.set_params(**self.best_params_)

        self.best_estimator_.fit(X, y)

        

        return self

    

    def predict(self, X):

        return self.best_estimator_.predict(X)

    

    def predict_proba(self, X):

        return self.best_estimator_.predict_proba(X)
model = lgb.LGBMClassifier()



# Set only upper and lower bounds for each parameter

param_grid = {

    'learning_rate': (0.01, 1),

    'n_estimators': (10, 200),

    'num_leaves': (2, 50),

    'reg_alpha': (0, 1),

    'reg_lambda': (0, 10),

    'min_split_gain': (0, 1),

}



model_bayes = BayesianSearchCV(

    model, param_grid, cv=5, n_iter=60, scoring='accuracy',

    int_parameters=['n_estimators', 'num_leaves'])



model_bayes.fit(X_train, y_train)
get_best_score(model_bayes);
predictions = model_default.predict(X_test)



submission = pd.DataFrame()

submission['PassengerId'] = PassengerId

submission['Survived'] = predictions

submission.to_csv('default.csv',index=False)



submission.head()
predictions = model_grid.predict(X_test)



submission = pd.DataFrame()

submission['PassengerId'] = PassengerId

submission['Survived'] = predictions

submission.to_csv('grid.csv',index=False)



submission.head()
predictions = model_rand.predict(X_test)



submission = pd.DataFrame()

submission['PassengerId'] = PassengerId

submission['Survived'] = predictions

submission.to_csv('rand.csv',index=False)



submission.head()
predictions = model_bayes.predict(X_test)



submission = pd.DataFrame()

submission['PassengerId'] = PassengerId

submission['Survived'] = predictions

submission.to_csv('bayes.csv',index=False)



submission.head()
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()



cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')



print('Average accuracy: ', cv_scores.mean())
model = DecisionTreeClassifier()



# Set only upper and lower bounds for each parameter

param_grid = {

    'max_depth': (2, 16),

    'min_samples_split': (2, 20),

    'min_samples_leaf': (1, 20),

    'max_leaf_nodes': (2, 100),

    'min_impurity_decrease': (0, 0.02),

    'ccp_alpha': (0, 0.01),

}



model_bayes = BayesianSearchCV(

    model, param_grid, cv=5, n_iter=60, scoring='accuracy',

    int_parameters=['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes'])



model_bayes.fit(X_train, y_train)



get_best_score(model_bayes);