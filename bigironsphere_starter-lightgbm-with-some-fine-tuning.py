import numpy as np 

import pandas as pd 

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

y = train['target'].values.flatten()

ids = train['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)

train.head()
print('train consists of {} rows and {} columns.'.format(train.shape[0], train.shape[1]))

print('train contains {} missing values.'.format(train.isna().sum().sum()))

print('train is {}% incomplete.'.format(100*train.isna().sum().sum()/(train.shape[0]*train.shape[1])))
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



N_FOLD = 5

folds = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)



oof = np.zeros(len(train))

importances = np.zeros(train.shape[1])

X = train.values

preds = np.zeros(len(test))



for train_idx, valid_idx in folds.split(X, y):



    X_train, X_valid = X[train_idx, :], X[valid_idx, :]

    y_train, y_valid = y[train_idx], y[valid_idx]

    

    model = lgb.LGBMClassifier(n_estimators=10000, eval_metric='auc')

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=200,

                      early_stopping_rounds=250, eval_metric='auc')

    val_preds = model.predict(X_valid)

    importances += model.feature_importances_/N_FOLD

    oof[valid_idx] = val_preds



AUC_OOF = round(roc_auc_score(y, oof), 4)

print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))
import matplotlib.pyplot as plt

import seaborn as sns



#make lgb feature importance df

feature_df = pd.DataFrame({'feature' : train.columns,

                             'importance' : importances})

feature_df = feature_df.sort_values('importance', ascending=False)

    

#plot feature importances

N_ROWS = 50

plot_height = int(np.floor(len(feature_df.iloc[:N_ROWS, :])/5))

plt.figure(figsize=(12, plot_height));

sns.barplot(x='importance', y='feature', data=feature_df.iloc[:N_ROWS, :]);

plt.title('LightGBM Feature Importance');

plt.show()
cards = []

for i in range(0, train.shape[1]):

    cards.append(len(np.unique(train.iloc[:, i].values)))

cards = np.asarray(cards)



card_df = pd.DataFrame({'feature' : train.columns,

                       'cardinality' : cards})



card_df.sort_values('cardinality', inplace=True)

card_df.head()        
temp = pd.concat([train['wheezy-copper-turtle-magic'], test['wheezy-copper-turtle-magic']])

temp = pd.factorize(temp)[0]

train['wheezy-copper-turtle-magic'] = temp[:len(train)]

test['wheezy-copper-turtle-magic'] = temp[len(train):]

train['wheezy-copper-turtle-magic'] = train['wheezy-copper-turtle-magic'].astype('category')

test['wheezy-copper-turtle-magic'] = test['wheezy-copper-turtle-magic'].astype('category')

cat_feature_index = [train.columns.get_loc('wheezy-copper-turtle-magic')]
oof = np.zeros(len(train))

importances = np.zeros(train.shape[1])

X = train.values



for train_idx, valid_idx in folds.split(X, y):



    X_train, X_valid = X[train_idx, :], X[valid_idx, :]

    y_train, y_valid = y[train_idx], y[valid_idx]

    

    model = lgb.LGBMClassifier(n_estimators=10000, eval_metric='auc')

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=200,

                      early_stopping_rounds=250, eval_metric='auc', categorical_feature=cat_feature_index)

    val_preds = model.predict(X_valid)

    importances += model.feature_importances_/N_FOLD

    oof[valid_idx] = val_preds



AUC_OOF = round(roc_auc_score(y, oof), 4)

print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))
#import required packages

from hyperopt import hp, tpe, Trials, STATUS_OK

from hyperopt.fmin import fmin

from hyperopt.pyll.stochastic import sample

import gc #garbage collection

#optional but advised

import warnings

warnings.filterwarnings('ignore')



#GLOBAL HYPEROPT PARAMETERS

NUM_EVALS = 1000 #number of hyperopt evaluation rounds

N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round



#LIGHTGBM PARAMETERS

LGBM_MAX_LEAVES = 2**9 #maximum number of leaves per tree for LightGBM

LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM

EVAL_METRIC_LGBM_REG = 'mae' #LightGBM regression metric. Note that 'rmse' is more commonly used 

EVAL_METRIC_LGBM_CLASS = 'auc'#LightGBM classification metric



def quick_hyperopt(data, labels, num_evals=NUM_EVALS, Class=True, cat_features=None):

    

    print('Running {} rounds of LightGBM parameter optimisation:'.format(num_evals))

    #clear space

    gc.collect()



    integer_params = ['max_depth',

                     'num_leaves',

                      'max_bin',

                     'min_data_in_leaf',

                     'min_data_in_bin']



    def objective(space_params):



        #cast integer params from float to int

        for param in integer_params:

            space_params[param] = int(space_params[param])



        #extract nested conditional parameters

        if space_params['boosting']['boosting'] == 'goss':

            top_rate = space_params['boosting'].get('top_rate')

            other_rate = space_params['boosting'].get('other_rate')

            #0 <= top_rate + other_rate <= 1

            top_rate = max(top_rate, 0)

            top_rate = min(top_rate, 0.5)

            other_rate = max(other_rate, 0)

            other_rate = min(other_rate, 0.5)

            space_params['top_rate'] = top_rate

            space_params['other_rate'] = other_rate



        subsample = space_params['boosting'].get('subsample', 1.0)

        space_params['boosting'] = space_params['boosting']['boosting']

        space_params['subsample'] = subsample



        if Class:                

            if cat_features is not None:

                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=True, categorical_feature=cat_features,

                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_CLASS, seed=42)

                best_loss = 1 - cv_results['auc-mean'][-1]

            else:

                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=True,

                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_CLASS, seed=42)

                best_loss = 1 - cv_results['auc-mean'][-1]



        else:

            if cat_features is not None:

                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=False,

                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)

                best_loss = cv_results['l1-mean'][-1] #'l2-mean' for rmse

            else:

                cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=True,

                                    early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)

                best_loss = 1 - cv_results['auc-mean'][-1]



        return{'loss':best_loss, 'status': STATUS_OK }



    if cat_features is not None:

        train = lgb.Dataset(data, labels, categorical_feature=cat_features)

    else:

         train = lgb.Dataset(data, labels)



    #integer and string parameters, used with hp.choice()

    boosting_list = [{'boosting': 'gbdt',

                      'subsample': hp.uniform('subsample', 0.5, 1)},

                     {'boosting': 'goss',

                      'subsample': 1.0,

                     'top_rate': hp.uniform('top_rate', 0, 0.5),

                     'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'



    if Class:

        metric_list = ['auc'] #modify as required for other classification metrics

        objective_list = ['binary', 'cross_entropy']



    else:

        metric_list = ['MAE', 'RMSE'] 

        objective_list = ['huber', 'gamma', 'fair', 'tweedie']





    space ={'boosting' : hp.choice('boosting', boosting_list),

            'num_leaves' : hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),

            'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),

            'max_bin': hp.quniform('max_bin', 32, 255, 1),

            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),

            'min_data_in_bin': hp.quniform('min_data_in_bin', 1, 256, 1),

            'min_gain_to_split' : hp.quniform('min_gain_to_split', 0.1, 5, 0.01),

            'lambda_l1' : hp.uniform('lambda_l1', 0, 5),

            'lambda_l2' : hp.uniform('lambda_l2', 0, 5),

            'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),

            'metric' : hp.choice('metric', metric_list),

            'objective' : hp.choice('objective', objective_list),

            'feature_fraction' : hp.quniform('feature_fraction', 0.5, 1, 0.01),

            'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1, 0.01)

        }



    trials = Trials()

    best = fmin(fn=objective,

                space=space,

                algo=tpe.suggest,

                max_evals=num_evals, 

                trials=trials)



    #fmin() will return the index of values chosen from the lists/arrays in 'space'

    #to obtain actual values, index values are used to subset the original lists/arrays

    best['boosting'] = boosting_list[best['boosting']]['boosting']#nested dict, index twice

    best['metric'] = metric_list[best['metric']]

    best['objective'] = objective_list[best['objective']]



    #cast floats of integer params to int

    for param in integer_params:

        best[param] = int(best[param])



    print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')

    return(best)    
lgbm_params = quick_hyperopt(train, y, 25, cat_features=cat_feature_index)
oof = np.zeros(len(train))

X = train.values

preds = np.zeros(len(test))



for train_idx, valid_idx in folds.split(X, y):



    X_train, X_valid = X[train_idx, :], X[valid_idx, :]

    y_train, y_valid = y[train_idx], y[valid_idx]

    

    train_dataset = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_index)

    val_dataset = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_feature_index)



    model = lgb.train(lgbm_params, train_dataset, valid_sets=[train_dataset, val_dataset], verbose_eval=200,

                      num_boost_round=10000, early_stopping_rounds=250, categorical_feature=cat_feature_index)

    val_preds = model.predict(X_valid, num_iteration=model.best_iteration)

    oof[valid_idx] = val_preds

    preds += model.predict(test, num_iteration=model.best_iteration)/N_FOLD



AUC_OOF = round(roc_auc_score(y, oof), 4)

print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('initial_sub.csv', index=False)
keep_cols = train.columns

remove_words = ['noise', 'distraction', 'discard']

for keyword in remove_words:

    keep_cols = [x for x in keep_cols if keyword not in keep_cols]

    

train_2 = train[keep_cols]

test_2 = test[keep_cols]

cat_feature_index = [train_2.columns.get_loc('wheezy-copper-turtle-magic')]



#get new optimised parameters

final_params = quick_hyperopt(train_2, y, 25, cat_features=cat_feature_index)



oof = np.zeros(len(train))

X = train_2.values

preds = np.zeros(len(test))



for train_idx, valid_idx in folds.split(X, y):



    X_train, X_valid = X[train_idx, :], X[valid_idx, :]

    y_train, y_valid = y[train_idx], y[valid_idx]

    

    train_dataset = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_index)

    val_dataset = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_feature_index)



    model = lgb.train(final_params, train_dataset, valid_sets=[train_dataset, val_dataset], verbose_eval=200,

                      num_boost_round=10000, early_stopping_rounds=250, categorical_feature=cat_feature_index)

    val_preds = model.predict(X_valid, num_iteration=model.best_iteration)

    oof[valid_idx] = val_preds

    preds += model.predict(test_2, num_iteration=model.best_iteration)/N_FOLD



AUC_OOF = round(roc_auc_score(y, oof), 4)

print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))
keep_cols = train.columns

remove_words = ['important', 'grandmaster', 'expert']

for keyword in remove_words:

    keep_cols = [x for x in keep_cols if keyword not in keep_cols]

    

train_3 = train[keep_cols]

test_3 = test[keep_cols]

cat_feature_index = [train_3.columns.get_loc('wheezy-copper-turtle-magic')]



#get new optimised parameters

final_params = quick_hyperopt(train_3, y, 25, cat_features=cat_feature_index)



oof = np.zeros(len(train))

X = train_3.values

preds = np.zeros(len(test))



for train_idx, valid_idx in folds.split(X, y):



    X_train, X_valid = X[train_idx, :], X[valid_idx, :]

    y_train, y_valid = y[train_idx], y[valid_idx]

    

    train_dataset = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_index)

    val_dataset = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_feature_index)



    model = lgb.train(final_params, train_dataset, valid_sets=[train_dataset, val_dataset], verbose_eval=200,

                      num_boost_round=10000, early_stopping_rounds=250, categorical_feature=cat_feature_index)

    val_preds = model.predict(X_valid, num_iteration=model.best_iteration)

    oof[valid_idx] = val_preds

    preds += model.predict(test_3, num_iteration=model.best_iteration)/N_FOLD



AUC_OOF = round(roc_auc_score(y, oof), 4)

print('Model ensemble OOF AUC score: {}'.format(AUC_OOF))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('second_sub.csv', index=False)