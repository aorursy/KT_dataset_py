import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
#Read files
train = pd.read_csv('../input/older-dataset-for-dont-overfit-ii-challenge/train.csv')
test = pd.read_csv('../input/older-dataset-for-dont-overfit-ii-challenge/test.csv')

#Check shapes
print('Train Shape: ', train.shape) 
print('Test Shape: ', test.shape)

train.head() #First 5 rows.
y_train = train['target'] #Assign the y target value for potential models.
X_train = train.drop(['target', 'id'], axis=1) #Drop target and ID for X_train

X_test = test.drop(['id'], axis=1) #Drop ID from the test set.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
log = linear_model.LogisticRegression(solver='liblinear')
ridge = linear_model.Ridge()
SGD = linear_model.SGDRegressor()
elastic = linear_model.ElasticNet()
lars = linear_model.Lars()
lasso = linear_model.Lasso()
lassolars = linear_model.LassoLars()
ortho = linear_model.OrthogonalMatchingPursuit()
ARD = linear_model.ARDRegression()
baye = linear_model.BayesianRidge()
def cv_scores(model):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc') #5 folds
    print('Model: ', model)
    print('CV Mean: ', np.mean(scores)) #Mean of the 5 scores
    print('STD: ', np.std(scores)) #Standard deviation of the 5 scores
    print('\n')
models = [log, ridge, SGD, elastic, lars, lasso, lassolars, ortho, ARD, baye]

for model in models:
    cv_scores(model)
n_nonzero_coefs = np.arange(1, 50, 1)
tol = [None, 1, 2, 5, 8, 15, 25, 35]
fit_intercept = [True, False]
normalize = [True, False]
precompute = [True, False]

from sklearn.model_selection import StratifiedKFold

parameters = dict(n_nonzero_coefs = n_nonzero_coefs,
             tol = tol,
             fit_intercept = fit_intercept,
             normalize = normalize,
             precompute = precompute)

grid = GridSearchCV(estimator = ortho, param_grid = parameters, scoring = 'roc_auc', verbose = 1, n_jobs=-1) #n_jobs use all proccessors
gridresult = grid.fit(X_train, y_train)

print('The best score was {:.5f} with parameters of {}'.format(gridresult.best_score_, gridresult.best_params_))
ortho = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs = 8, fit_intercept = True, normalize = False, precompute = True,
                                              tol = None)
cv_scores(ortho)
predict = ortho.fit(X_train, y_train).predict(X_test)

submission = pd.read_csv('../input/older-dataset-for-dont-overfit-ii-challenge/sample_submission.csv')
submission['target'] = predict
submission.to_csv('submission.csv', index=False)

submission.head()
solver = ['liblinear', 'saga'] #both handle l1 and l2 penalty
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
class_weight = ['balanced']

parameters = dict(solver = solver,
             penalty = penalty,
             C = C,
             class_weight = class_weight)

grid = GridSearchCV(estimator = log, param_grid = parameters, scoring = 'roc_auc', verbose = 1, n_jobs=-1) #n_jobs use all proccessors
gridresult = grid.fit(X_train, y_train)

print('The best score was {:.5f} with parameters of {}'.format(gridresult.best_score_, gridresult.best_params_))
log = linear_model.LogisticRegression(C = 1, class_weight = {1: 0.5, 0: 0.5}, penalty = 'l1', solver = 'liblinear')
cv_scores(log)
predict = log.fit(X_train, y_train).predict(X_test)

submission = pd.read_csv('../input/older-dataset-for-dont-overfit-ii-challenge/sample_submission.csv')
submission['target'] = predict
submission.to_csv('submission.csv', index=False)

submission.head()
n_iter = np.arange(1,501,100)
verbose = [True, False]
alpha_1 = (1e-9, 1.0, 'log-uniform')
alpha_2 = (1e-9, 1.0, 'log-uniform')
lambda_1 = (1e-9, 1000, 'log-uniform')
lambda_2 = (1e-9, 1000, 'log-uniform')

parameters = dict(n_iter = n_iter,
                 verbose = verbose,
                 alpha_1 = alpha_1,
                 alpha_2 = alpha_2,
                 lambda_1 = lambda_1,
                 lambda_2 = lambda_2)

grid = GridSearchCV(estimator = ARD, param_grid = parameters, scoring = 'roc_auc', verbose = 1, n_jobs=-1) #n_jobs use all proccessors
gridresult = grid.fit(X_train, y_train)

print('The best score was {:.5f} with parameters of {}'.format(gridresult.best_score_, gridresult.best_params_))

#randomsearch = RandomizedSearchCV(estimator = ARD, param_distributions = parameters, scoring = 'roc_auc', verbose = 1, 
                                  #n_jobs= -1)
#searchresult = randomsearch.fit(X_train, y_train)

#print('The best score was {:.5f} with parameters of {}'.format(searchresult.best_score_, searchresult.best_params_))
ARD = linear_model.ARDRegression(alpha_1=1e-09, alpha_2 = 1.0, lambda_1 = 1e-09, lambda_2 = 1e-09, n_iter = 1, verbose = True)

cv_scores(ARD)
predict = ARD.fit(X_train, y_train).predict(X_test)

submission = pd.read_csv('../input/older-dataset-for-dont-overfit-ii-challenge/sample_submission.csv')
submission['target'] = predict
submission.to_csv('submission.csv', index=False)

submission.head()
from skopt import BayesSearchCV

n_iter = np.arange(1,501,100)
alpha_1 = (1e-9, 1.0, 'log-uniform')
alpha_2 = (1e-9, 1.0, 'log-uniform')
lambda_1 = (1e-9, 1000, 'log-uniform')
lambda_2 = (1e-9, 1000, 'log-uniform')


params = dict(n_iter = n_iter,
             alpha_1 = alpha_1,
             alpha_2 = alpha_2,
             lambda_1 = lambda_1,
             lambda_2 = lambda_2,
             )

#bayes = BayesSearchCV(estimator = baye, search_spaces = params, scoring='roc_auc', verbose=1, n_jobs=-1, n_iter=12)
#bayesresult = bayes.fit(X_train, y_train)
#print('The best score was {:.5f} with parameters of {}'.format(bayesresult.best_score_, bayesresult.best_params_))

grid = GridSearchCV(estimator = baye, param_grid = params, scoring='roc_auc', verbose=1, n_jobs=-1)
gridresult = grid.fit(X_train, y_train)
print('The best score was {:.5f} with parameters of {}'.format(gridresult.best_score_, gridresult.best_params_))
baye = linear_model.BayesianRidge(alpha_1=1.0, alpha_2=1.0, lambda_1=1e-09, lambda_2=1e-09, n_iter=1)
cv_scores(baye)
predict = baye.fit(X_train, y_train).predict(X_test)

submission = pd.read_csv('../input/older-dataset-for-dont-overfit-ii-challenge/sample_submission.csv')
submission['target'] = predict
submission.to_csv('submission.csv', index=False)

submission.head()
penalty = ['l1', 'l2', 'elasticnet']
alpha = [1, 10, 100, 1000]
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
eta0 = [1, 10, 100]

params = dict(
                           penalty=penalty,
                           alpha=alpha,
                           learning_rate=learning_rate,
                           eta0=eta0)

grid = GridSearchCV(estimator = SGD, param_grid = params, scoring='roc_auc', verbose=1, n_jobs=-1)
gridresult = grid.fit(X_train, y_train)
print('The best score was {:.5f} with parameters of {}'.format(gridresult.best_score_, gridresult.best_params_))
SGD = linear_model.SGDRegressor(alpha=10, eta0=1, learning_rate='adaptive', penalty='l2')
cv_scores(SGD)
predict = SGD.fit(X_train, y_train).predict(X_test)

submission = pd.read_csv('../input/older-dataset-for-dont-overfit-ii-challenge/sample_submission.csv')
submission['target'] = predict
submission.to_csv('submission.csv', index=False)

submission.head()
models = []
sub_score = []

models = ['Orthogonal Matching', 'Linear Regression', 'ARD Regression', 'Bayesian Ridge', 'SGD Regressor']
sub_score = [0.839, 0.735, 0.733, 0.740, 0.746]

for i in range(len(models)):
    print(models[i], 'with a score of: ', sub_score[i])
