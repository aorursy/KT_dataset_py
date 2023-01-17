import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import pickle as pkl

import sys

import seaborn as sns



from fastFM import als

from fastFM.datasets import make_user_item_regression

from scipy.sparse import csc_matrix

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder

from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

from hyperopt import tpe, fmin, hp, Trials, STATUS_OK



% matplotlib inline

sns.set_style('whitegrid')
# import data

data_path = os.path.join('..','..', 'data-2')

splits_path = os.path.join(data_path,'splits')

sparse_path = os.path.join(data_path, 'sparse')

columns = ['user','item','rating']



with open(os.path.join(splits_path, 'train.df'), 'rb') as file_in:

    train_df = pkl.load(file_in)

    

with open(os.path.join(splits_path, 'dev.df'), 'rb') as file_in:

    cv_df = pkl.load(file_in)

    

with open(os.path.join(splits_path, 'test.df'), 'rb') as file_in:

    test_df = pkl.load(file_in)



# import sparse data and store as a dictionary for easy access

sparse = dict()

features = ['actors', 'country', 'directors-imdb', 'genres-amazon', 'genres-imdb', 'language', 'mpaa',

           'studios-amazon', 'studios-imdb', 'type', 'user-item']

for feature in features:

    with open(os.path.join(sparse_path, feature + '.dict'), 'rb') as file_in:

        sparse[feature] = pkl.load(file_in)
X_train = sparse['user-item']['train']

X_cv = sparse['user-item']['cv']

X_test = sparse['user-item']['test']



y_train = train_df['rating']

y_cv = cv_df['rating']

y_test = test_df['rating']
# define the objective function that the fmin module can later optimize on

def test_fm(params):

    

    # convert certain hyperparameter choices to integers

    print('==========TESTING FM==========')

    params['n_iter'] = int(params['n_iter'])

    params['rank'] = int(params['rank'])

    print(params)

    

    # train model and evaluate MAE on cross-validation set

    fm = als.FMRegression(**params)

    y_train = train_df['rating']

    fm.fit(X_train, y_train)

    y_cv_pred = fm.predict(X_cv)

    y_cv = cv_df['rating'].values

    

    mae = mean_absolute_error(y_cv_pred, y_cv)

    print('MAE:', mae)

    return mae
use_pretrained = True



if use_pretrained:

    with open(os.path.join(data_path, 'trials_fm_ratings_only'), 'rb') as file_in:

        trials = pkl.load(file_in)

    with open(os.path.join(data_path, 'best_fm_ratings_only.dict'), 'rb') as file_in:

        best = pkl.load(file_in)

else:

    # set hyperparameter ranges

    trials = Trials()

    space = {

        'n_iter': hp.uniform('n_iter', 100, 1000), # number of parameter updates using ALS

        'init_stdev': hp.uniform('init_stdev', 0, 1), # standard deviation of initialized parameters

        'rank': hp.uniform('rank', 2, 5), # rank of the factorization used for the second order interactions

        'l2_reg_w': hp.uniform('l2_reg_w', 0, 11), # L2 penalty weight for linear coefficients

        'l2_reg_V': hp.uniform('l2_reg_V', 0, 11), # L2 penalty weight for pairwise coefficients

    }

    

    # Choose the Tree-structured Parzen Estimator (TPE) as the algorithm to optimize the objective function

    best = fmin(algo = tpe.suggest,

               fn = test_fm,

               trials = trials,

               max_evals = 100, # max number of tests

               space = space)

    with open(os.path.join(data_path, 'trials_fm_ratings_only'), 'wb') as file_out:

        pkl.dump(trials, file_out)

    with open(os.path.join(data_path, 'best_fm_ratings_only.dict'), 'wb') as file_out:

        pkl.dump(best, file_out)
def val_diagnostic(val_name, trials):

    '''generates scatter plot and histogram of each hyperparameter, as well as a plot on loss values (MAE)'''

    

    vals = np.array([trial['misc']['vals'][val_name] for trial in trials.trials])

    

    # convert certain hyperparameter values to integers

    if val_name in ['n_iter', 'rank']:

        vals = [int(x) for x in vals]

        

    ts = [trial['tid'] for trial in trials.trials]

    results = [trial['result']['loss'] for trial in trials.trials]

    

    fig, axes = plt.subplots(1, 3, figsize = (16,4))

    axes[0].scatter(ts, vals)

    axes[0].set(xlabel='iteration', ylabel=val_name)

    axes[1].hist(np.array(vals).squeeze())

    axes[1].set(xlabel=val_name, ylabel='frequency')

    axes[2].scatter(vals, results)

    axes[2].set(xlabel=val_name, ylabel='loss')

    plt.tight_layout()
for val in trials.trials[0]['misc']['vals'].keys():

    val_diagnostic(val, trials)
# optimized hyperparameter values

best_opt = best.copy()

best_opt['n_iter'] = int(best_opt['n_iter'])

best_opt['rank'] = int(best_opt['rank'])

best_opt
use_pretrained = True



if use_pretrained:

    train_results_df = pd.read_csv(os.path.join(data_path, 'results_fm_ratings_train.csv'))

    cv_results_df = pd.read_csv(os.path.join(data_path, 'results_fm_ratings_cv.csv'))

    test_results_df = pd.read_csv(os.path.join(data_path, 'results_fm_ratings_test.csv'))

else:

    fm = als.FMRegression(**best_opt)

    fm.fit(X_train, y_train)

    

    y_train_pred = fm.predict(X_train)

    train_results_df = train_df[['user','item','rating']].copy()

    train_results_df['prediction'] = y_train_pred

    train_results_df.to_csv(os.path.join(data_path, 'results_fm_ratings_train.csv'), header=True, index=False)

    

    y_cv_pred = fm.predict(X_cv)

    cv_results_df = cv_df[['user','item','rating']].copy()

    cv_results_df['prediction'] = y_cv_pred

    cv_results_df.to_csv(os.path.join(data_path, 'results_fm_ratings_cv.csv'), header=True, index=False)

    

    y_test_pred = fm.predict(X_test)

    test_results_df = test_df[['user','item','rating']].copy()

    test_results_df['prediction'] = y_test_pred

    test_results_df.to_csv(os.path.join(data_path, 'results_fm_ratings_test.csv'), header=True, index=False)
# Some predictions exceed the permitted range of 1 to 5.

# All predicted values less than 1 are converted 1, and all predicted values greater than 5 are converted to 5.

y_train_pred = np.clip(train_results_df['prediction'], a_min=1, a_max=5)

y_cv_pred = np.clip(cv_results_df['prediction'], a_min=1, a_max=5)

y_test_pred = np.clip(test_results_df['prediction'], a_min=1, a_max=5)



# Calculate and print MAE

mae_train = np.abs(train_results_df['rating'] - y_train_pred).mean()

mae_cv = np.abs(cv_results_df['rating'] - y_cv_pred).mean()

mae_test = np.abs(test_results_df['rating'] - y_test_pred).mean()



print('mean absolute error, training set:', mae_train)

print('mean absolute error, cross-validation set:', mae_cv)

print('mean absolute error, test set:', mae_test)



mae_list = [mae_train, mae_cv, mae_test]



# plot results

plt.figure(figsize = (10, 5))

ax = plt.barh([0,1,2], mae_list, alpha = 0.7)

plt.gca().set(yticks = [0,1,2], yticklabels = ['train', 'cross-validation', 'test'])

plt.gca().set(title = 'Factorization machine with only ratings data: mean absolute error by data set',

              ylabel = 'Dataset', xlabel = 'Mean absolute error')
def calc_auc(data):

    y = data['rating'] >= 4

    y_pred = np.clip(data['prediction'], a_min=1, a_max=5)

    y_pred = y_pred / y_pred.max()

    auc = roc_auc_score(y, y_pred)

    return auc



auc_train = calc_auc(train_results_df)

auc_cv = calc_auc(cv_results_df)

auc_test = calc_auc(test_results_df)



print('ROC AUC, training set:', auc_train)

print('ROC AUC, cross-validation set:', auc_cv)

print('ROC AUC, test set:', auc_test)



auc_list = [auc_train, auc_cv, auc_test]



plt.figure(figsize = (10, 5))

ax = plt.barh([0,1,2], auc_list, alpha = 0.7)

plt.gca().set(yticks = [0,1,2], yticklabels = ['train', 'cross-validation', 'test'])

plt.gca().set(title = 'Factorization machine with only ratings data: ROC area under curve by data set',

              ylabel = 'Dataset', xlabel = 'AUC')