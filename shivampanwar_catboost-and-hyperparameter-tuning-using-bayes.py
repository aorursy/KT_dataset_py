import os

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn import metrics

import numpy as np

import pandas as pd

from time import time

import pprint

import joblib

import warnings

warnings.filterwarnings("ignore")



# Classifiers

from catboost import CatBoostClassifier



# Model selection

from sklearn.model_selection import StratifiedKFold



# Metrics

from sklearn.metrics import roc_auc_score

from sklearn.metrics import make_scorer



# Skopt functions

from skopt import BayesSearchCV

from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper

from skopt.space import Real, Categorical, Integer

from time import time

PATH_TO_DATA = '../input/'



df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'train_targets.csv'), 

                                   index_col='match_id_hash')

df_train_features.head(n=2)
df_train_features.shape
y_target=df_train_targets['radiant_win'].apply(lambda x:int(x)).values
X_train, X_valid, y_train, y_valid = train_test_split(df_train_features.values, y_target, 

                                                      test_size=0.3, 

                                                      random_state=17)
%%time

base_model = CatBoostClassifier(task_type = "GPU",verbose=True)

base_model.fit(X_train,y_train,)
y_pred = base_model.predict_proba(X_valid)[:, 1]

valid_score = roc_auc_score(y_valid, y_pred)

print('Validation ROC-AUC score:', valid_score)
def submit_predictions(name,model):

    df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')

    X_test = df_test_features.values

    y_test_pred = model.predict_proba(X_test)[:, 1]



    df_submission = pd.DataFrame({'radiant_win_prob': y_test_pred}, 

                                 index=df_test_features.index)

    submission_filename = 'submission_{}.csv'.format(name)

    df_submission.to_csv(submission_filename)

    print('Submission saved to {}'.format(submission_filename))
submit_predictions('simple_cat_boost',model=base_model)
# Reporting util for different optimizers

def report_perf(optimizer, X, y, title, callbacks=None):

    """

    A wrapper for measuring time and performances of different optmizers

    

    optimizer = a sklearn or a skopt optimizer

    X = the training set 

    y = our target

    title = a string label for the experiment

    """

    start = time()

    if callbacks:

        optimizer.fit(X, y, callback=callbacks)

    else:

        optimizer.fit(X, y)

    d=pd.DataFrame(optimizer.cv_results_)

    best_score = optimizer.best_score_

    best_score_std = d.iloc[optimizer.best_index_].std_test_score

    best_params = optimizer.best_params_

    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "

           +u"\u00B1"+" %.3f") % (time() - start, 

                                  len(optimizer.cv_results_['params']),

                                  best_score,

                                  best_score_std))    

    print('Best parameters:')

    pprint.pprint(best_params)

    print()

    return best_params
roc_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
clf = CatBoostClassifier(thread_count=2,

                         loss_function='Logloss',

                        

                         od_type = 'Iter',

                         verbose= False

                        )
# Defining your search space

search_spaces = {'iterations': Integer(10, 1000),

                 'depth': Integer(1, 8),

                 'learning_rate': Real(0.01, 1.0, 'log-uniform'),

                 'random_strength': Real(1e-9, 10, 'log-uniform'),

                 'bagging_temperature': Real(0.0, 1.0),

                 'border_count': Integer(1, 255),

                 'l2_leaf_reg': Integer(2, 30),

                 'scale_pos_weight':Real(0.01, 1.0, 'uniform')}

# Setting up BayesSearchCV

opt = BayesSearchCV(clf,

                    search_spaces,

                    scoring=roc_auc,

                    cv=skf,

                    n_iter=100,

                    n_jobs=1,  # use just 1 job with CatBoost in order to avoid segmentation fault

                    return_train_score=False,

                    refit=True,

                    optimizer_kwargs={'base_estimator': 'GP'},

                    random_state=42)
# best_params = report_perf(opt, df_train_features, y_target,'CatBoost', 

#                           callbacks=[VerboseCallback(100), 

#                                      DeadlineStopper(60*10)])
best_params={'bagging_temperature': 0.41010395885331385,

 'border_count': 186,

 'depth': 8,

 'iterations': 323,

 'l2_leaf_reg': 21,

 'learning_rate': 0.0673344419215237,

 'random_strength': 3.230824361824754e-06,

 'scale_pos_weight': 0.7421091918485163}
best_params['iterations']=1000
# %%time

# tuned_model = CatBoostClassifier(**best_params,task_type = "GPU",od_type='Iter',one_hot_max_size=10)

# tuned_model.fit(X_train,y_train)
# %%time

# y_pred = tuned_model.predict_proba(X_valid)[:, 1]

# valid_score = roc_auc_score(y_valid, y_pred)

# print('Validation ROC-AUC score:', valid_score)

# ##ROC_AUC is 0.8056
%%time

tuned_model = CatBoostClassifier(**best_params,task_type = "GPU",od_type='Iter',one_hot_max_size=10)

tuned_model.fit(df_train_features,y_target)
submit_predictions('tuned_cat_boost',model=tuned_model)