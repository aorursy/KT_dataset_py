!pip install git+https://github.com/scikit-optimize/scikit-optimize.git --user
import datetime

import os



import numpy as np

import pandas as pd

import pprint



# % matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.metrics import make_scorer

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split



from catboost import CatBoostClassifier

import random



# Skopt functions

from skopt import BayesSearchCV

from skopt.callbacks import DeadlineStopper, VerboseCallback

from skopt.space import Real, Integer

from time import time
# Reporting util for different optimizers

def report_perf(optimizer, x, y, title, callbacks=None):

    """

    A wrapper for measuring time and performances of different optmizers



    optimizer = a sklearn or a skopt optimizer

    X = the training set

    y = our target

    title = a string label for the experiment

    """

    start = time()

    if callbacks:

        optimizer.fit(x, y, callback=callbacks)

    else:

        optimizer.fit(x, y)

    d = pd.DataFrame(optimizer.cv_results_)

    best_score = optimizer.best_score_

    best_score_std = d.iloc[optimizer.best_index_].std_test_score

    best_params = optimizer.best_params_

    print((

                  title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "

                  + u"\u00B1" + " %.3f") % (time() - start,

                                            len(optimizer.cv_results_[

                                                    'params']),

                                            best_score,

                                            best_score_std))

    print('Best parameters:')

    pprint.pprint(best_params)

    print()

    return best_params
RANDOM_STATE = 1242

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)



NUM_THREADS = 2

NUM_THREADS_FILE = 4

NUM_THREADS_PRED = 2

# def read_data(file_path):

print('Loading datasets...')

file_path = '../input/cat-in-the-dat-ii/'

train = pd.read_csv(file_path + 'train.csv', sep=',')

test = pd.read_csv(file_path + 'test.csv', sep=',')

print('Datasets loaded')

# return train, test

# train, test = read_data(PATH)



print(train.shape, test.shape)

print(train.head(2))

print(test.head(2))



X = train.drop(['id', 'target'], axis=1)

categorical_features = [col for c, col in enumerate(X.columns)

                        if not (np.issubdtype(X.dtypes[c], np.number))]

y = train['target']



print(len(categorical_features), X.shape, y.shape, y.mean())

X = X.fillna(-65500)

for f in categorical_features:

    X[f] = X[f].astype('category')



X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2,

                                  random_state=RANDOM_STATE, stratify=y)

print(X1.shape, X2.shape, y1.shape, y2.shape, y1.mean(), y2.mean(), y.mean())
roc_auc = make_scorer(roc_auc_score, greater_is_better=True,

                      needs_threshold=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

clf = CatBoostClassifier(thread_count=NUM_THREADS,

                         loss_function='Logloss',

                         cat_features=categorical_features,

#                          scale_pos_weight = (len(y1)-sum(y1))/sum(y1),

                         od_type='Iter',

                         nan_mode='Min',

                         early_stopping_rounds=300,

                         iterations=1000,

#                          subsample=0.5,

                         eval_metric='AUC',

                         metric_period=50,

                         task_type='GPU',

                         verbose=False

                         )



# Defining your search space

search_spaces = {  # 'iterations': Integer(10, 1000),

    'depth': Integer(1, 5),

    'learning_rate': Real(0.02, 0.6, 'log-uniform'),

    #                  'random_strength': Real(1e-9, 10, 'log-uniform'),

    'random_strength': Integer(1, 30000000),

    #                  'rsm': Real(0.1, 1.0), #cpu only

    'bagging_temperature': Real(0.1, 2.0),

#     'one_hot_max_size': Integer(2, 15),

    'border_count': Integer(10, 255),

    'min_data_in_leaf': Integer(5, 1000), #gpu only

    'l2_leaf_reg': Integer(5, 1500000),

    #                  'scale_pos_weight':Real(0.01, 1.0, 'uniform')

}

# Setting up BayesSearchCV

opt = BayesSearchCV(clf,

                    search_spaces,

                    scoring=roc_auc,

                    cv=skf,

                    n_iter=100,

                    n_jobs=1,# use just 1 job with CatBoost in order to avoid segmentation fault

                    return_train_score=False,

                    refit=True,

                    optimizer_kwargs={'base_estimator': 'GP'},

                    random_state=RANDOM_STATE)
best_params = report_perf(opt, X2, y2, 'CatBoost', # for faster work using x2

                          callbacks=[VerboseCallback(100),

                                     DeadlineStopper(60 * 60 * 3)])  # 3 hours
print(best_params)
print('Start fit.', datetime.datetime.now() )



# params from: https://www.kaggle.com/lucamassaron/catboost-in-action-with-dnn



# best_params = {'bagging_temperature': 0.8,

#                'depth': 5,

#                'iterations': 50000,

#                'l2_leaf_reg': 30,

#                'learning_rate': 0.05,

#                'random_strength': 0.8}



model = CatBoostClassifier(**best_params,

                           loss_function='Logloss',

                           eval_metric='AUC',

                           nan_mode='Min',

                           thread_count=4, task_type='GPU',

                           verbose=False)



model.fit(X1, y1, eval_set=(X2, y2), cat_features=categorical_features,

          verbose_eval=300,

          early_stopping_rounds=500,

          use_best_model=True,

          plot=True)



pred = model.predict_proba(X2)[:, 1]

score = roc_auc_score(y2, pred)

print(score)

print('End fit.', datetime.datetime.now())

X_test = test.drop('id', axis=1)

X_test = X_test.fillna(-65500)

for f in categorical_features:

    X_test[f] = X_test[f].astype('category')



pd.DataFrame(

    {'id': test['id'], 'target': model.predict_proba(X_test)[:, 1]}).to_csv(

    'submission.csv', index=False)
