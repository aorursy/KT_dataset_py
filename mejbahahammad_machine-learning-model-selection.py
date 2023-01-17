

# Load libraries

import numpy as np

import warnings

warnings.filterwarnings('ignore')

from sklearn import linear_model, datasets

from sklearn.model_selection import GridSearchCV

# Load data

iris = datasets.load_iris()

features = iris.data

target = iris.target

# Create logistic regression

logistic = linear_model.LogisticRegression()

# Create range of candidate penalty hyperparameter values

penalty = ['l1', 'l2']

# Create range of candidate regularization hyperparameter values

C = np.logspace(0, 4, 10)

# Create dictionary hyperparameter candidates

hyperparameters = dict(C=C, penalty=penalty)



# Create grid search

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)



# Fit grid search

best_model = gridsearch.fit(features, target)

np.logspace(0, 4, 10)





print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])

print('Best C:', best_model.best_estimator_.get_params()['C'])



# Predict target vector

best_model.predict(features)
# Load libraries

from scipy.stats import uniform

from sklearn import linear_model, datasets

from sklearn.model_selection import RandomizedSearchCV

# Load data

iris = datasets.load_iris()

features = iris.data

target = iris.target

# Create logistic regression

logistic = linear_model.LogisticRegression()

# Create range of candidate regularization penalty hyperparameter values

penalty = ['l1', 'l2']

# Create distribution of candidate regularization hyperparameter values

C = uniform(loc=0, scale=4)

# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)

# Create randomized search

randomizedsearch = RandomizedSearchCV(

logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,

n_jobs=-1)

# Fit randomized search

best_model = randomizedsearch.fit(features, target)



# Define a uniform distribution between 0 and 4, sample 10 values

uniform(loc=0, scale=4).rvs(10)



# View best hyperparameters

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])

print('Best C:', best_model.best_estimator_.get_params()['C'])



# Predict target vector

best_model.predict(features)
# Load libraries

import numpy as np

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

# Set random seed

np.random.seed(0)

# Load data

iris = datasets.load_iris()

features = iris.data

target = iris.target

# Create a pipeline

pipe = Pipeline([("classifier", RandomForestClassifier())])

# Create dictionary with candidate learning algorithms and their hyperparameters

search_space = [{"classifier": [LogisticRegression()],

"classifier__penalty": ['l1', 'l2'],

"classifier__C": np.logspace(0, 4, 10)},

{"classifier": [RandomForestClassifier()],

"classifier__n_estimators": [10, 100, 1000],

"classifier__max_features": [1, 2, 3]}]

# Create grid search

gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)

# Fit grid search

best_model = gridsearch.fit(features, target)



# View best model

best_model.best_estimator_.get_params()["classifier"]



# Predict target vector

best_model.predict(features)
# Load libraries

import numpy as np

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

# Set random seed

np.random.seed(0)

# Load data

iris = datasets.load_iris()

features = iris.data

target = iris.target

# Create a preprocessing object that includes StandardScaler features and PCA

preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# Create a pipeline

pipe = Pipeline([("preprocess", preprocess),

("classifier", LogisticRegression())])

# Create space of candidate values

search_space = [{"preprocess__pca__n_components": [1, 2, 3],

"classifier__penalty": ["l1", "l2"],

"classifier__C": np.logspace(0, 4, 10)}]

# Create grid search

clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)

# Fit grid search

best_model = clf.fit(features, target)



# View best model

best_model.best_estimator_.get_params()['preprocess__pca__n_components']
# Load libraries

import numpy as np

from sklearn import linear_model, datasets

from sklearn.model_selection import GridSearchCV

# Load data

iris = datasets.load_iris()

features = iris.data

target = iris.target



# Create logistic regression

logistic = linear_model.LogisticRegression()

# Create range of candidate regularization penalty hyperparameter values

penalty = ["l1", "l2"]

# Create range of candidate values for C

C = np.logspace(0, 4, 1000)

# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)

# Create grid search

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)

# Fit grid search

best_model = gridsearch.fit(features, target)



# Create grid search using one core

clf = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=1, verbose=1)

# Fit grid search

best_model = clf.fit(features, target)
# Load libraries

from sklearn import linear_model, datasets

# Load data

iris = datasets.load_iris()

features = iris.data

target = iris.target

# Create cross-validated logistic regression

logit = linear_model.LogisticRegressionCV(Cs=100)

# Train model

logit.fit(features, target)
# Load libraries

import numpy as np

from sklearn import linear_model, datasets

from sklearn.model_selection import GridSearchCV, cross_val_score

# Load data

iris = datasets.load_iris()

features = iris.data

target = iris.target

# Create logistic regression

logistic = linear_model.LogisticRegression()

# Create range of 20 candidate values for C

C = np.logspace(0, 4, 20)

# Create hyperparameter options

hyperparameters = dict(C=C)

# Create grid search



gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)

# Conduct nested cross-validation and outut the average score

cross_val_score(gridsearch, features, target).mean()



gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1)



best_model = gridsearch.fit(features, target)



scores = cross_val_score(gridsearch, features, target)