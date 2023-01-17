# installing the keras tuner library

!pip install -q -U keras-tuner
# importing the dependencies

import pandas as pd

import kerastuner as kt

from sklearn import ensemble

from sklearn import datasets

from sklearn import linear_model

from sklearn import metrics

from sklearn import model_selection

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer



#loading the dataset

data = load_breast_cancer()



data = load_breast_cancer()

df = pd.DataFrame(data['data'], columns=data['feature_names'])

df['target'] = data['target']

df.info()

df.head(3)
# displaying the target names

list(data.target_names)
# Selecting few features, for the sake of simplicity

X = df[['mean radius', 'worst concave points', 'worst area', 

          'mean concavity', 'mean concave points']]

y = df[['target']]
# spliting the dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
def build_random_forest(hp):

  model = ensemble.RandomForestClassifier(

        n_estimators=hp.Int('n_estimators', 10, 50, step=10),

        max_depth=hp.Int('max_depth', 3, 10))

  return model
tuner = kt.tuners.Sklearn(

    oracle=kt.oracles.BayesianOptimization(

        objective=kt.Objective('score', 'max'),

        max_trials=10),

    hypermodel= build_random_forest,

    directory='.',

    project_name='random_forest')
tuner.search(X_train.values, y_train.values.ravel())

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hp)
model.fit(X_train.values, y_train.values.ravel())

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
def build_DecisionTreeClassifier(hp):

    model = DecisionTreeClassifier(max_depth=hp.Int("max_depth", 50, 100, step=10))

    return model
tuner = kt.tuners.Sklearn(

    oracle=kt.oracles.BayesianOptimization(

        objective=kt.Objective('score', 'max'),

        max_trials=10),

    hypermodel= build_DecisionTreeClassifier,

    directory='.',

    project_name='DecisionTreeClassifier')
model = tuner.hypermodel.build(best_hp)
model.fit(X_train.values, y_train.values.ravel())

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
def build_ridge_classifier(hp):

  model = linear_model.RidgeClassifier(

        alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))

  return model
tuner = kt.tuners.Sklearn(

    oracle=kt.oracles.BayesianOptimization(

        objective=kt.Objective('score', 'max'),

        max_trials=10),

    hypermodel= build_ridge_classifier,

    directory='.',

    project_name='ridge_classifier')
tuner.search(X_train.values, y_train.values.ravel())

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hp)

model.fit(X_train.values, y_train.values.ravel())
predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
def build_svc(hp):

  model = SVC(C=hp.Float('c', 1.0, 20.0, step=10),

              gamma=hp.Float('gamma', 0.01, 1, step=0.01)

              )

  return model
tuner = kt.tuners.Sklearn(

    oracle=kt.oracles.BayesianOptimization(

        objective=kt.Objective('score', 'max'),

        max_trials=10),

    hypermodel= build_svc,

    directory='.',

    project_name='svc')
tuner.search(X_train.values, y_train.values.ravel())

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hp)



model.fit(X_train.values, y_train.values.ravel())

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
def build_model(hp):

  model_type = hp.Choice('model_type', ['random_forest', 'ridge', 'SVC'])

  if model_type == 'random_forest':

    model = ensemble.RandomForestClassifier(

        n_estimators=hp.Int('n_estimators', 10, 50, step=10),

        max_depth=hp.Int('max_depth', 3, 10))

  elif model_type == 'ridge':

    model = linear_model.RidgeClassifier(

        alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))

  else:

    model = SVC(C=hp.Float('c', 1.0, 20.0, step=10),

              gamma=hp.Float('gamma', 0.01, 1, step=0.01))

  return model


tuner = kt.tuners.Sklearn(

    oracle=kt.oracles.BayesianOptimization(

        objective=kt.Objective('score', 'max'),

        max_trials=10),

    hypermodel=build_model,

    scoring=metrics.make_scorer(metrics.accuracy_score),

    cv=model_selection.StratifiedKFold(5),

    directory='.',

    project_name='my_best_model')
tuner.search(X_train.values, y_train.values.ravel())

best_model = tuner.get_best_models(num_models=1)[0]
best_model.fit(X_train.values, y_train.values.ravel())

predictions = best_model.predict(X_test)

print(accuracy_score(y_test, predictions))