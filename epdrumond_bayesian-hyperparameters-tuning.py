#Basic imports

import pandas as pd

import numpy as np

from tqdm import tqdm

import time



#Plotting imports

import matplotlib.pyplot as plt

import seaborn as sns



#General Sklearn imports

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV



#Hyperopt imports

import hyperopt as hp

from hyperopt import fmin, tpe, Trials, hp, STATUS_OK, space_eval

from hyperopt.pyll.base import scope



#Classification imports

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
#Ignore warnings

import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")
#Load dataset

data = pd.read_csv('../input/heart-disease-uci/heart.csv')
#Encode multiclass categorical features

to_encode = ['cp', 'restecg', 'slope', 'ca', 'thal']

for col in to_encode:

    new_feat = pd.get_dummies(data[col])

    new_feat.columns = [col+str(x) for x in range(new_feat.shape[1])]

    new_feat = new_feat.drop(columns = new_feat.columns.values[0])

    

    data[new_feat.columns.values] = new_feat

    data = data.drop(columns = col)
#Split data into attributes and target

atr = data.drop(columns = 'target')

target = data['target']
#Scale dataset

scaler = MinMaxScaler()

atr = scaler.fit_transform(atr)
#Preliminar modeling

pre_score = cross_val_score(estimator = GaussianNB(),

                            X = atr, 

                            y = target,

                            scoring = 'accuracy',

                            cv = 10,

                            verbose = 0)



print('Naive-Bayes mean score: %5.3f' %np.mean(pre_score))
#Compare algorithms in their default configurations

models = [LogisticRegression(), DecisionTreeClassifier(), SVC()]

model_names = [type(x).__name__ for x in models]



std_score = []

for m in tqdm(models):

    std_score.append(cross_val_score(estimator = m,

                                 X = atr,

                                 y = target,

                                 scoring = 'accuracy',

                                 cv = 10).mean())

    

pd.Series(data = std_score, index = model_names)
#Bayesian hyperparameters tuning: Define function

def bayes_tuning(estimator, xdata, ydata, cv, space, max_it):

    

    #Define objective function

    def obj_function(params):

        model = estimator(**params)

        score = cross_val_score(estimator = model, X = xdata, y = ydata,

                                scoring = 'accuracy',

                                cv = cv).mean()

        return {'loss': -score, 'status': STATUS_OK}

    

    start = time.time()

    

    #Perform tuning

    hist = Trials()

    param = fmin(fn = obj_function, 

                 space = space,

                 algo = tpe.suggest,

                 max_evals = max_it,

                 trials = hist,

                 rstate = np.random.RandomState(1))

    param = space_eval(space, param)

    

    #Compute best score

    score = -obj_function(param)['loss']

    

    return param, score, hist, time.time() - start
#Define hyperparameters spaces for Bayesian tuning

lr_params = {'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2)),

             'C': hp.uniform('C', 0.1, 2.0),

             'fit_intercept': hp.choice('fit_intercept', [True, False]),

             'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),

             'max_iter': scope.int(hp.quniform('max_iter', 50, 500, 20))

}



dt_params = {'criterion': hp.choice('criterion', ['gini', 'entropy']),

             'splitter': hp.choice('splitter', ['best', 'random']),

             'max_depth': scope.int(hp.quniform('max_depth', 3, 50, 1)),

             'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 50, 1)),

             'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 50, 1)),

             'max_features': hp.choice('max_features', ['auto', 'log2', None])

}



sv_params = {'C': hp.uniform('C', 0.1, 2.0),

             'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),

             'degree': scope.int(hp.quniform('degree', 2, 5, 1)),

             'gamma': hp.choice('gamma', ['auto', 'scale']),

             'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2)),

             'max_iter': scope.int(hp.quniform('max_iter', -1, 100, 1))

}
#Apply bayesian tuning

models = [LogisticRegression, DecisionTreeClassifier, SVC]

model_params = [lr_params, dt_params, sv_params]



bayes_score, bayes_time, bayes_hist = [], [], []

for m, par in tqdm(zip(models, model_params)):

    param, score, hist, dt = bayes_tuning(m, atr, target, 10, par, 150)

    bayes_score.append(score)

    bayes_time.append(dt)

    bayes_hist.append(hist)
#Print bayesian tuning results

bayes_df = pd.DataFrame(index = model_names)

bayes_df['Accuracy'] = bayes_score

bayes_df['Time'] = bayes_time



print(bayes_df)
#Define function for grid search tuning

def grid_tuning(estimator, xdata, ydata, cv, space):

    

    start = time.time()

    

    #Perform tuning

    grid = GridSearchCV(estimator = estimator,

                        param_grid = space,

                        scoring = 'accuracy',

                        cv = 10)

    grid.fit(xdata, ydata)

    

    return grid.best_params_, grid.best_score_, time.time() - start
#Define hyperparameters spaces for grid seach tuning

lr_params = {'tol': [1e-5, 1e-3, 1e-2],

             'C': [0.1, 0.5, 1.0, 2.0],

             'fit_intercept': [True, False],

             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

             'max_iter': [50, 100, 250, 500]

}



dt_params = {'criterion': ['gini', 'entropy'],

             'splitter': ['best', 'random'],

             'max_depth': [3, 10, 25, 40, 50],

             'min_samples_split': [2, 10, 25, 50, 50],

             'min_samples_leaf': [1, 10, 25, 50, 50],

             'max_features': ['auto', 'log2', None]

}



sv_params = {'C': [0.1, 0.5, 1.0, 2.0],

             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

             'degree': [2, 3, 5],

             'gamma': ['auto', 'scale'],

             'tol': [1e-5, 1e-3, 1e-2],

             'max_iter': [-1, 50, 100]

}
#Apply grid seach tuning

models = [LogisticRegression(), DecisionTreeClassifier(), SVC()]

model_params = [lr_params, dt_params, sv_params]



grid_score, grid_time = [], []

for m, par in tqdm(zip(models, model_params)):

    _, score, dt = grid_tuning(m, atr, target, 10, par)

    grid_score.append(score)

    grid_time.append(dt)
#Print grid search tuning results

grid_df = pd.DataFrame(index = model_names)

grid_df['Accuracy'] = grid_score

grid_df['Time'] = grid_time



print(grid_df)
#Define function for random search tuning

def random_tuning(estimator, xdata, ydata, cv, space, max_iter):

    

    start = time.time()

    

    #Perform tuning

    rand = RandomizedSearchCV(estimator = estimator,

                              param_distributions = space,

                              n_iter = max_iter,

                              scoring = 'accuracy',

                              cv = 10,

                              random_state = np.random.RandomState(1))

    rand.fit(xdata, ydata)

    

    return rand.best_params_, rand.best_score_, rand.cv_results_['mean_test_score'], time.time() - start
#Define hyperparameters spaces for random search tuning

lr_params = {'tol': list(np.logspace(np.log(1e-5), np.log(1e-2), num = 10, base = 3)),

             'C': list(np.linspace(0.1, 2.0, 20)),

             'fit_intercept': [True, False],

             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

             'max_iter': list(range(50, 501))

}



dt_params = {'criterion': ['gini', 'entropy'],

             'splitter': ['best', 'random'],

             'max_depth': list(range(3, 51)),

             'min_samples_split': list(range(2, 50)),

             'min_samples_leaf': list(range(1, 50)),

             'max_features': ['auto', 'log2', None]

}



sv_params = {'C': list(np.linspace(0.1, 2.0, 10)),

             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

             'degree': list(range(2, 6)),

             'gamma': ['auto', 'scale'],

             'tol': list(np.logspace(np.log(1e-5), np.log(1e-2), num = 10, base = 3)),

             'max_iter': list(range(-1, 101))

}
#Apply random seach tuning

models = [LogisticRegression(), DecisionTreeClassifier(), SVC()]

model_params = [lr_params, dt_params, sv_params]



rand_score, rand_time, rand_hist = [], [], []

for m, par in tqdm(zip(models, model_params)):

    _, score, hist, dt = random_tuning(m, atr, target, 10, par, 150)

    rand_score.append(score)

    rand_time.append(dt)

    rand_hist.append(hist)
#Print random search tuning results

rand_df = pd.DataFrame(index = model_names)

rand_df['Accuracy'] = rand_score

rand_df['Time'] = rand_time



print(rand_df)
#Install plotly

!pip install plotly
#Plotly imports

import plotly.graph_objects as go

from plotly.subplots import make_subplots
#Compare accuracy

tuning_acc = pd.DataFrame(index = model_names)

tuning_acc['Bayes'] = bayes_score

tuning_acc['Grid'] = grid_score

tuning_acc['Random'] = rand_score



fig = go.Figure(data = [

    go.Bar(name = 'Bayes Tuning', x = tuning_acc.index, y = tuning_acc['Bayes']),

    go.Bar(name = 'Grid Tuning', x = tuning_acc.index, y = tuning_acc['Grid']),

    go.Bar(name = 'Random Tuning', x = tuning_acc.index, y = tuning_acc['Random'])

])



fig.update_layout(barmode = 'group', 

                  title = 'Accuracy Comparison',

                  xaxis_title = 'Estimator',

                  yaxis_title = 'Cross-validation accuracy (%)',

                  yaxis = dict(range = [0.75, 0.9]))

fig.show()
#Compare performance

tuning_time = pd.DataFrame(index = model_names)

tuning_time['Bayes'] = bayes_time

tuning_time['Grid'] = grid_time

tuning_time['Random'] = rand_time



fig = go.Figure(data = [

    go.Bar(name = 'Bayes Tuning', x = tuning_time.index, y = tuning_time['Bayes']),

    go.Bar(name = 'Grid Tuning', x = tuning_time.index, y = tuning_time['Grid']),

    go.Bar(name = 'Random Tuning', x = tuning_time.index, y = tuning_time['Random'])

])



fig.update_layout(barmode = 'group',

                  title = 'Performance Comparison',

                  xaxis_title = 'Estimator',

                  yaxis_title = 'Computation time (sec)')

fig.show()
#Compare Bayesian and Random

bayes_best = dict()

random_best = dict()

for i,model in enumerate(model_names):

    dummy = [-x['loss'] for x in bayes_hist[i].results]

    bayes_best[model] = np.maximum.accumulate(dummy)

    

    dummy = [x for x in rand_hist[i]]

    random_best[model] = np.maximum.accumulate(dummy)
#Logistic Regression

fig = go.Figure()



fig.add_trace(go.Scatter(x = list(range(150)), y = bayes_best['LogisticRegression'], name = 'Bayes (Hyperopt)'))

fig.add_trace(go.Scatter(x = list(range(150)), y = random_best['LogisticRegression'], name = 'Random search'))



fig.update_layout(title = 'Logistic Regression Tuning progression',

                  xaxis_title = 'Iteration',

                  yaxis_title = 'Cross-validation accuracy (%)')

fig.show()
#DecisionTreeClassifier

fig = go.Figure()



fig.add_trace(go.Scatter(x = list(range(150)), y = bayes_best['DecisionTreeClassifier'], name = 'Bayes (Hyperopt)'))

fig.add_trace(go.Scatter(x = list(range(150)), y = random_best['DecisionTreeClassifier'], name = 'Random search'))



fig.update_layout(title = 'Decision Tree Tuning progression',

                  xaxis_title = 'Iteration',

                  yaxis_title = 'Cross-validation accuracy (%)')

fig.show()
#SVC

fig = go.Figure()



fig.add_trace(go.Scatter(x = list(range(150)), y = bayes_best['SVC'], name = 'Bayes (Hyperopt)'))

fig.add_trace(go.Scatter(x = list(range(150)), y = random_best['SVC'], name = 'Random search'))



fig.update_layout(title = 'SVC Tuning progression',

                  xaxis_title = 'Iteration',

                  yaxis_title = 'Cross-validation accuracy (%)')

fig.show()