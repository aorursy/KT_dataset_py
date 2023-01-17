# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/exams/Exams"))
train = pd.read_csv("../input/exams/Exams/train.gz")

test = pd.read_csv("../input/exams/Exams/test.gz")

y_test = pd.read_csv("../input/exams/Exams/y_test.gz")
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

                    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
train.head()
test.head()
train.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(17,10))

sns.countplot(x="click",data=train)

print(train.click.value_counts())
#Đếm số unique của cột 0 và cột 1 ở tập train

print(len(train['0'].unique()))

print(len(train['1'].unique()))
#Đếm số unique của cột 0 và cột 1 ở tập test

print(len(test['0'].unique()))

print(len(test['1'].unique()))
#Giả thuyết cột 0 và 1 là `ID`, đếm số lượng ID trong tập test mà có trong tập train

print(len(set(test['0'].unique()) - set(train['0'].unique())))

print(len(set(test['1'].unique()) - set(train['1'].unique())))
for i in ['2','3','4','5','6']:

    print('value counts of col %s'%(i))

    print(train[i].value_counts())
fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(17,10))

sns.countplot(x="2", hue="click", data=train, ax=axs[0][0])

sns.countplot(x="3", hue="click", data=train, ax=axs[0][1])

sns.countplot(x="4", hue="click", data=train, ax=axs[1][0])

sns.countplot(x="6", hue="click", data=train, ax=axs[1][1])
plt.figure(figsize=(17,10))

sns.countplot(x="5", hue="click", data=train)
num_cols = [str(i) for i in range(11,35)]

corr_df = train[num_cols].corr()

plt.figure(figsize=(17,10))

sns.heatmap(corr_df)
sampled_train = pd.concat([train.loc[train['click'] == 0].sample(500),

          train.loc[train['click'] == 1].sample(500)])
sns.pairplot(sampled_train, 

             hue='click',

            vars=num_cols)

plt.show()
import xgboost as xgb
from sklearn.metrics import roc_auc_score,classification_report,auc,roc_curve,precision_recall_curve

import time

import datetime

def fast_auc(y_true, y_prob):

    """

    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    nfalse = 0

    auc = 0

    n = len(y_true)

    for i in range(n):

        y_i = y_true[i]

        nfalse += (1 - y_i)

        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))

    return auc





def eval_auc(y_true, y_pred):

    """

    Fast auc eval function for lgb.

    """

    return 'auc', fast_auc(y_true, y_pred), True
from imblearn.over_sampling import (RandomOverSampler, 

                                    SMOTE, 

                                    ADASYN)

def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):

    """

    A function to train a variety of regression models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    

    """

    columns = X.columns if columns == None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'AUC',

                        'sklearn_scoring_function': roc_auc_score},

                    }

    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros((len(X), len(set(y.values))))

    

    # averaged predictions on train data

    prediction = np.zeros((len(X_test), oof.shape[1]))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        sampler = SMOTE(sampling_strategy='minority')

        X_train, y_train = sampler.fit_sample(X_train,y_train)

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_test)

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid

        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid[:, 1]))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict
y = train.click

X = train.drop(['click','4'],axis=1)

X_test = test.drop(['4'],axis=1)
from sklearn.preprocessing import LabelEncoder

for f in X.columns:

    if X[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(X[f].values) + list(X_test[f].values))

        X[f] = lbl.transform(list(X[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values)) 
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit



n_fold = 5

# folds = TimeSeriesSplit(n_splits=n_fold)

folds = KFold(n_splits=5)
import lightgbm as lgb

params = {'num_leaves': 256,

          'min_child_samples': 79,

          'objective': 'binary',

          'max_depth': 13,

          'learning_rate': 0.03,

          "boosting_type": "gbdt",

          "subsample_freq": 3,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3,

          'reg_lambda': 0.3,

          'colsample_bytree': 0.9

         }
result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=False,

                                                      verbose=100, early_stopping_rounds=20, n_estimators=500)
y_pred_prob = result_dict_lgb['prediction'][:,1]
y_pred = [1 if i >0.5 else 0 for i in y_pred_prob]

print(classification_report(y_test,y_pred))
def plot_roc_curve(y_true,y_pred_prob):

    plt.figure()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    roc_auc = auc(fpr, tpr) # compute area under the curve

    plt.figure(figsize=(17,10))

    plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % (roc_auc))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    ax2 = plt.gca().twinx()

    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')

    ax2.set_ylabel('Threshold',color='r')

    ax2.set_ylim([thresholds[-1],thresholds[0]])

    ax2.set_xlim([fpr[0],fpr[-1]])
plot_roc_curve(y_test, y_pred_prob)
def plot_prec_recall_thresh(y_true,y_pred_prob):

    prec, rec, thresholds = precision_recall_curve(y_true,y_pred_prob)



    plt.figure(figsize=(17,10))

    plt.plot(thresholds, prec[:-1], 'b--', label='precision')

    plt.plot(thresholds, rec[:-1], 'g--', label = 'recall')

    plt.xlabel('Threshold')

    plt.legend(loc='upper left')

    plt.ylim([0,1])



plot_prec_recall_thresh(y_test, y_pred_prob)

plt.show()
from sklearn.metrics import precision_recall_curve, f1_score, auc, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# calculate F1 score

f1 = f1_score(y_test, y_pred)

# calculate precision-recall AUC

auc = auc(recall, precision)

# calculate average precision score

ap = average_precision_score(y_test, y_pred_prob)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))

# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.')

# show the plot

plt.show()