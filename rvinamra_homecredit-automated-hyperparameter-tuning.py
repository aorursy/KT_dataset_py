# Will Koehrsen's notebook used as a reference

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb # modelling

# evaluating the model

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

# visualization

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 18
%matplotlib inline

# governing choices for search

N_FOLDS = 5
MAX_EVALS = 5

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# immporting datasets and its splitting

features = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

# sampling 16000 rows (10000 for training, 6000 for testing)

features = features.sample(n = 16000, random_state = 42)

# selecting only numeric features

features = features.select_dtypes('number')

# extracting labels

labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1, ))
features = features.drop(columns = ['TARGET', 'SK_ID_CURR'])

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 6000, random_state = 42)

print('Train shape: ', train_features.shape)
print('Test shape: ', test_features.shape)

train_features.head()
model = lgb.LGBMClassifier(random_state = 50)

# training set
train_set = lgb.Dataset(train_features, label = train_labels)
test_set = lgb.Dataset(test_features, label = test_labels)
# default hyperparameters
hyperparameters = model.get_params()

# using early stopping to determine number of estimators and therefore deleting the default value
del hyperparameters['n_estimators']

# performing cross validation with early stopping
cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=10000, nfold = N_FOLDS, metrics = 'auc', early_stopping_rounds=100, verbose_eval=False, seed = 42)

# highest score
best = cv_results['auc-mean'][-1]

# standard deviation of best score
best_std = cv_results['auc-stdv'][-1]

print('The maximium ROC AUC in cross validation was {:.5f} with std of {:.5f}.'.format(best, best_std))
print('The ideal number of iterations was {}.'.format(len(cv_results['auc-mean'])))
# evaluating the baseline model on the testing data
# optimal number of estimators found in cv

model.n_estimators = len(cv_results['auc-mean'])

# training and making predictions with model
model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(test_labels, preds)

print('The baseline model scores {:.5f} ROC-AUC on the test set.'.format(baseline_auc) )
# Objective Function which takes a set of Hyperparameter values and returns the cross validation score on the training data

# the following excerpt about objective function in Hyperopt taken from Will Koehrsen's notebook for future reference
# An objective function in Hyperopt must return either a single real value to minimize, or a dictionary with a key "loss" 
# with the score to minimize (and a key "status" indicating if the run was successful or not).

# Optimization is typically about minimizing a value, and because our metric is Receiver Operating Characteristic Area Under the Curve (ROC AUC) 
# where higher is better, the objective function will return  1−ROC AUC Cross Validation

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

def objective(hyperparameters):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.
       Writes a new line to `outfile` on every iteration
    """
    
    # keeping track of evals
    global ITERATION
    ITERATION += 1
    
    # using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']
    
    # retrieving the subsample
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)
    
    # extracting the boosting types and subsample to top level keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample
    
    # ensuring the integer parameters remain as intergers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])
        
    start = timer()
    
    # performing n_folds cv
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 1000, nfold = N_FOLDS, early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    
    run_time = timer() - start
    
    # extracting the best score
    best_score = cv_results['auc-mean'][-1]
    
    # minimizing loss
    loss = 1-best_score
    
    # boosting rounds that returned the highest score
    n_estimators = len(cv_results['auc-mean'])
    
    # adding the number of estimators to the hyperparameters
    hyperparameters['n_estimators'] = n_estimators 
    
    # writing to the csv file. 'a' mean append
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
    of_connection.close()
    
    # dictionary with information for evaluation
    return{'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION, 'train_time': run_time, 'status': STATUS_OK}
# specifying domain (known as space in Hyperopt) is different from grid search. In Hyperopt and other Bayesian opt. frameworks
# the domain is not a discrete grid but it consists of probability distribution of different hyperparameters

from hyperopt import hp
from hyperopt.pyll.stochastic import sample
# again like grid search we will go through example of learning rate which is defined on a log scale

# creating the learning rate
learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}
# visualization of the learning rate
learning_rate_dist = []

# drawing 1000 samples from the learning rate domain
for _ in range(10000):
    learning_rate_dist.append(sample(learning_rate)['learning_rate'])
    
plt.figure(figsize = (8,6))
sns.kdeplot(learning_rate_dist, color = 'red', linewidth = 2, shade = True);
plt.title('Learning Rate Dist.', size = 18); plt.xlabel('Learning Rate', size = 16); plt.ylabel('Density', size = 16);
# on the other, no. of leaves is an uniform distribution

num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}
num_leaves_dist = []

# sampling 10000 times from the number of leaves distribution
for _ in range(10000):
    num_leaves_dist.append(sample(num_leaves)['num_leaves'])
    
plt.figure(figsize = (8,6))
sns.kdeplot(num_leaves_dist, linewidth = 2, shade = True);
plt.title('Number of Leaves Distribution', size = 18); 
plt.xlabel('Number of Leaves', size = 16); plt.ylabel('Density', size = 16);
# conditional domain
# using nested conditional statements to indicate hyperparameters that depend on other hyperparameters
# for example, "goss" boosting_type cannot use subsampling, therefore we have to preset the "subsample" value as 1.0 in its case

# boosting type domain
boosting_type = {'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)}, 
                                                             {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                                             {'boosting_type': 'goss', 'subsample':1.0}])}

# drawing a sample
hyperparams = sample(boosting_type)
hyperparams
# retrieve the subsample if present otherwise set it to a default of 1.0
subsample = hyperparams['boosting_type'].get('subsample', 1.0)

# extracting the boosting type
hyperparams['boosting_type'] = hyperparams['boosting_type']['boosting_type']
hyperparams['subsample'] = subsample

hyperparams
# the gbm cannot use the distionary, therefore 'boosting_type' and 'subsample' have to be set as toplevel keys

# COMPLETE BAYESIAN DOMAIN
# defining the search space

space = {
    'boosting_type': hp.choice('boosting_type', 
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
}
# sampling from the full space (domain)
x = sample(space)

# conditional logic to assign top-level keys
subsample = x['boosting_type'].get('subsample', 1.0)
x['boosting_type'] = x['boosting_type']['boosting_type']
x['subsample'] = subsample

# will different outputs each time its run
x
# testing objective function with domain
# creating a new file and opening a connection

OUT_FILE = 'bayes_test.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

ITERATION = 0

# writing column names

headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()

# testing the objective function
# results vary everytime
results = objective(sample(space))
print('The cross validation loss = {:.5f}'.format(results['loss']))
print('The optimal number of estimators was {}.'.format(results['hyperparameters']['n_estimators']))


# OPTIMIZATION ALGORITHM

from hyperopt import tpe

# creating the algorithm
tpe_algorithm = tpe.suggest
# automated hyperparameter tuning is informed unlike grid and random methods
# hyperopt internally keeps a track of the results for the algos to use, but if we want to monitor the results and have a saved copy of the search, storage of results is required
# Trials object stores the dictionary returned from the objective function

from hyperopt import Trials

# recording results
trials = Trials()
# Trials object will hold everything returned from objective functions in the .results attribute

# Create a file and open a connection
OUT_FILE = 'bayes_test.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

ITERATION = 0

# Write column names
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()
# now we have all the four parts of the hyperparameter optimization

from hyperopt import fmin 
# fmin takes the four parts defined above as well as the maximum number of iterations max_evals

global ITERATION

ITERATION = 0

# running optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials, max_evals = MAX_EVALS)

best
# the best object holds only the hyperparameter which returned the lowest loss function

# for understanding how each search progresses, inspection of Trials object or csv file is required

# sorting the trials with the lowest loss (highest AUC) first
trials_dict = sorted(trials.results, key = lambda x: x['loss'])
trials_dict[:1]
# reading the csv file
results = pd.read_csv(OUT_FILE)
# this function takes in the results, trains a model on the training data and evaluates on the testing data
# returns a df of hyperparameters from the search. 
# saving the results to a csv file converts the dictionary of hyperparameters to a string
# mapping that back to a dictionary using ast.literal_eval

import ast

def evaluate(results, name):
    """
    Evaluate model on test data using hyperparameters in results
    Return dataframe of hyperparameters
    """
    new_results = results.copy()
    # string to dictionary
    new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)
    
    # sorting with the best values on top
    new_results = new_results.sort_values('score', ascending = False).reset_index(drop = True)
    
    # printing out cross validation high score
    print('The highest cross validation score from {} was {:.5f} found on iteration {}.'.format(name, new_results.loc[0, 'score'], new_results.loc[0, 'iteration']))
    
    # using the best parameters to create a model
    hyperparameters = new_results.loc[0, 'hyperparameters']
    model = lgb.LGBMClassifier(**hyperparameters)
    
    # training and making predictions
    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]
    
    print('ROC AUC from {} on test data = {:.5f}.'.format(name, roc_auc_score(test_labels, preds)))
    
    # creating dataframes of hyperparameters
    hyp_df = pd.DataFrame(columns = list(new_results.loc[0, 'hyperparameters'].keys()))
    
    # iterating through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(new_results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index = [0]), ignore_index = True)
        
    # putting the iteration and score in hyperparameter dataframe
    hyp_df['iteration'] = new_results['iteration']
    hyp_df['score'] = new_results['score']
    
    return hyp_df
bayes_results = evaluate(results, name = 'Bayesian')
bayes_results
# continuing optimization
# Hyperopt can continue searching where a previous search left off if we pass in a  Trials object that already has results
MAX_EVALS = 10

best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials, max_evals = MAX_EVALS)

# saving the Trials object so it can be realater for more training

import json

with open('trials.json', 'w') as f:
    f.write(json.dumps(trials_dict))
# to start the training from where it left off, simply load in the Trials object and pass it to an instance of fmin
MAX_EVALS = 300

# Create a new file and open a connection
OUT_FILE = 'bayesian_trials_300.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

# Write column names
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()

# Record results
trials = Trials()

global ITERATION

ITERATION = 0 

best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials, max_evals = MAX_EVALS)

# Sort the trials with lowest loss (highest AUC) first
trials_dict = sorted(trials.results, key = lambda x: x['loss'])

print('Finished, best results')
print(trials_dict[:1])

# Save the trial results
with open('trials.json', 'w') as f:
    f.write(json.dumps(trials_dict))
# going through the results from 300 search iterations on the reduced dataset. Looking at the scores, the distribution of hyperparameter values tried, 
# the evolution of values over time, and compare the hyperparameters values to those from random search.

bayes_results = pd.read_csv('../input/bayes-300/bayesian_trials_300.csv').sort_values('score', ascending = False).reset_index()
bayes_params = evaluate(bayes_results, name = 'Bayesian')
# getting all the score in a df in order to plot them over the course of training

scores = pd.DataFrame({'ROC AUC': bayes_params['score'], 'iteration': bayes_params['iteration'], 'search': 'Bayesian'})
scores['ROC AUC'] = scores['ROC AUC'].astype(np.float32)
scores['iteration'] = scores['iteration'].astype(np.int32)

scores.head()
# best scores for plotting the best hyperparameter values

best_bayes_params = bayes_params.iloc[bayes_params['score'].idxmax(), :].copy()
# plot of scores over the course of searching

sns.lmplot('iteration', 'ROC AUC', hue = 'search', data = scores, size = 8);
plt.scatter(best_bayes_params['iteration'], best_bayes_params['score'], marker = '*', s = 40, c = 'orange', edgecolors='k')
plt.xlabel('Iteration'); plt.ylabel('ROC AUC'); plt.title("Validation ROC AUC v/s Iteration")
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18

# density plots of the learning rate distribution
sns.kdeplot(learning_rate_dist, label = 'Sampling Distribution', linewidth = 4)
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayesian Optimization', linewidth = 4)
plt.legend()
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');
# we can make the same plot now for all the hyperparameters

# iterating through each hyperparameter
for i, hyper in enumerate(bayes_params.columns):
    if hyper not in ['class_weight', 'n_estimators', 'score', 'is_unbalance',
                    'boosting_type', 'iteration', 'subsample', 'metric', 'verbose', 'loss', 'learning_rate']:
        plt.figure(figsize = (14, 6))
        if hyper != 'loss':
            sns.kdeplot([sample(space[hyper]) for _ in range(1000)], label = 'Sampling Dist.', linewidth = 4)
            sns.kdeplot(bayes_params[hyper], label = 'Bayes Opt.', linewidth = 4)
            plt.legend()
            plt.title('{} Distribution'.format(hyper))
            plt.xlabel('{}'.format(hyper)); plt.ylabel('Density');
            plt.show();
# evolution of hyperparameters over search

fig, axs = plt.subplots(1, 4, figsize = (24, 6))
i=0

for i, hyper in enumerate(['colsample_bytree', 'learning_rate', 'min_child_samples', 'num_leaves']):
    sns.regplot('iteration', hyper, data = bayes_params, ax = axs[i])
    axs[i].scatter(best_bayes_params['iteration'], best_bayes_params[hyper], marker = '*', s = 20, c = 'k')
    axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = '{} over Search'.format(hyper));
    
plt.tight_layout()
# Read in full dataset
train = pd.read_csv('../input/data-will/simple_features_train.csv')
test = pd.read_csv('../input/homecred-final-test/simple_features_test.csv')

# Extract the test ids and train labels
test_ids = test['SK_ID_CURR']
train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))

train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])
test = test.drop(columns = ['SK_ID_CURR'])

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)
# APPLYING TO FULL DATASET
bayes_results['hyperparameters'] = bayes_results['hyperparameters'].map(ast.literal_eval)

# bayesian optimization on full dataset

hyperparameters = dict(**bayes_results.loc[0, 'hyperparameters'])
del hyperparameters['n_estimators']

# cross validation with n_folds and early stopping
cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=10000, early_stopping_rounds=100, metrics = 'auc', nfold = N_FOLDS)
print('The cross validation score on the full dataset for Bayes Opt. = {:.5f} with std: {:.5f}.'.format(cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
print('Number of estimators = {}.'.format(len(cv_results['auc-mean'])))
model = lgb.LGBMClassifier(n_estimators = len(cv_results['auc-mean']), **hyperparameters)
model.fit(train, train_labels)

preds = model.predict_proba(test)[:, 1]

submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})
submission.to_csv('submission_bayesian_optimization2.csv', index = False)