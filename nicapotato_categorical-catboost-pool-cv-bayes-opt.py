# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import catboost

from sklearn import preprocessing

from contextlib import contextmanager

import time

import seaborn as sns

import matplotlib.pyplot as plt

import gc



from bayes_opt import BayesianOptimization

import warnings



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from catboost.utils import get_gpu_device_count

print('\n%i GPU devices available' % get_gpu_device_count())



# Any results you write to the current directory are saved as output.

print(catboost.__version__)



notebookstart = time.time()

seed = 25



top_boosting_rounds = 9000

early_stopping_rounds = 100
@contextmanager

def timer(name):

    """

    Time Each Process

    """

    t0 = time.time()

    yield

    print('\n[{}] done in {} Minutes'.format(name, round((time.time() - t0)/60,2)))
with timer("Load"):

    nrow = None



    train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col = 'id', nrows = nrow)

    test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col = 'id')

    submission_df = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")

    [print(x.shape) for x in [train, test, submission_df]]



    traindex = train.index

    testdex = test.index



    y = train.target.copy()



    df = pd.concat([train.drop('target',axis = 1), test], axis = 0)

    del train, test, submission_df
with timer("Categorical Processing"):

    categorical = df.columns

    # Encoder:

    for col in categorical:

        diff = list(set(df.loc[testdex, col].unique()) - set(df.loc[traindex,col].unique()))

        if diff:

            print("Column {} has {} unseen categories in test set".format(col, len(diff)))

            df.loc[df[col].isin(diff),col] = 999

        if df[col].dtypes == object:

            df[col] = df[col].astype(str)

        lbl = preprocessing.LabelEncoder()

        df[col] = pd.Series(lbl.fit_transform(df[col].values)).astype('category')
# Prepare Data Object

categorical_index = list(range(0, len(categorical)))

features_names = df.columns



catboost_pool = catboost.Pool(df.loc[traindex,:],

    label=y,

    cat_features=categorical_index)



test_pool = catboost.Pool(data=df.loc[testdex,:],

    cat_features = categorical_index)



del df

gc.collect()
def catboost_blackbox(max_depth, reg_lambda):

    # num_leaves removed

    param = {

        'learning_rate': 0.2,

        'bagging_temperature': 0.1, 

        'l2_leaf_reg': reg_lambda,

        'depth': int(max_depth), 

#         'max_leaves': int(num_leaves),

#         'max_bin':255,

        'iterations' : top_boosting_rounds,

        'task_type':'GPU',

#         'grow_policy': 'Lossguide '

        'loss_function' : "Logloss",

        'objective':'Logloss',

        'eval_metric' : "AUC",

        'bootstrap_type' : 'Bayesian',

        'random_seed': seed,

        'early_stopping_rounds' : early_stopping_rounds,

        'use_best_model': False,

        "verbose": False

    }

    

    modelstart= time.time()

    scores = catboost.cv(catboost_pool,

                param,

                fold_count = 2,

                stratified = True,

                shuffle = True,

                partition_random_seed = seed,

                plot = False

                )

    runtime = (time.time() - modelstart)/60

    

    optimise = scores.loc[scores['test-AUC-mean'].idxmax(),'test-AUC-mean'] - scores.loc[scores['test-AUC-mean'].idxmax(),'test-AUC-std']

    optimisation_info.append([scores['test-AUC-mean'].idxmax(), optimise, runtime, param, scores['test-AUC-mean'].idxmax()])

    

    

    return optimise
parameter_bounds = {

#     'num_leaves': (31, 500), 

    'reg_lambda': (0.1, 10),

    'max_depth':(3,16)

}



init_points = 2

n_iter = 8



optimisation_info = []

CATBOOST_BO = BayesianOptimization(catboost_blackbox,

                                   parameter_bounds,

                                   random_state=seed)
with timer("Bayesian Optimisation - {} Iterations".format(init_points + n_iter)):

    with warnings.catch_warnings():

        warnings.filterwarnings('ignore')

        CATBOOST_BO.maximize(init_points = init_points,

                             n_iter = n_iter,

                             acq = 'ucb',

                             xi = 0.0,

                             alpha = 1e-6)
print("Best Score: {}".format(CATBOOST_BO.max['target']))
CATBOOST_BO.max['params']
optimisation_pd = pd.DataFrame(optimisation_info, columns = ['Best Round', 'Score', 'Runtime','Param', 'Iterations'])

optimisation_pd.head()
optimisation_pd.describe()
best_param = optimisation_pd.loc[optimisation_pd['Score'].idxmax(),'Param']

best_param['iterations'] = top_boosting_rounds*3

best_param['learning_rate'] = 0.04

best_param['early_stopping_rounds'] = early_stopping_rounds



best_param
with timer("Catboost CV"):

    scores = catboost.cv(catboost_pool,

                best_param,

                fold_count = 3,

                stratified = True,

                partition_random_seed = seed,

                plot = True,

                shuffle = True,

                )



display(scores.tail())



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].plot(scores['iterations'],scores['test-Logloss-mean'], label='test-Logloss-mean')

axes[0].plot(scores['iterations'],scores['train-Logloss-mean'], label='train-Logloss-mean')

axes[0].legend()



axes[1].plot(scores['iterations'],scores['test-AUC-mean'], label='validation_rocauc')

axes[1].legend()

plt.show()



best_iteration = scores['test-AUC-mean'].idxmax()

print("Best Iteration: {}".format(best_iteration))



display(scores.loc[best_iteration,:])
with timer("Catboost Single Model"):

    best_param['iterations'] = best_iteration

    model = catboost.CatBoostClassifier(**best_param)

    model.fit(catboost_pool)
feat_imp = pd.DataFrame()

feat_imp['importance'] = model.get_feature_importance()

feat_imp['features'] = features_names

feat_imp.sort_values(by = 'importance', inplace = True, ascending = False)



sns.barplot(y = feat_imp['features'], x = feat_imp['importance'])

plt.title("Feature Importance")

plt.show()
cm = catboost.utils.get_confusion_matrix(model, catboost_pool)

print(cm)
roc_curve_values = catboost.utils.get_roc_curve(model, catboost_pool, plot=True)
(thresholds, fnr) = catboost.utils.get_fnr_curve(curve=roc_curve_values, plot=True)
(thresholds, fpr) = catboost.utils.get_fpr_curve(curve=roc_curve_values, plot=True)
results = model.predict_proba(test_pool)[:, 1]

submission = pd.DataFrame({'id': testdex, 'target': results})

submission.to_csv('submission.csv', index=False)
!head submission.csv
print("Notebook Runtime: %0.2f Hours"%((time.time() - notebookstart)/60/60))