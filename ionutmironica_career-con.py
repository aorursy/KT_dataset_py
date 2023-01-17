import numpy as np 

import pandas as pd 

import os

from time import time

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from matplotlib import rcParams

%matplotlib inline

le = preprocessing.LabelEncoder()

from numba import jit

import itertools

from seaborn import countplot,lineplot, barplot

from numba import jit

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn import preprocessing

from scipy.stats import randint as sp_randint

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import LeaveOneGroupOut

from sklearn.model_selection import GroupKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



import matplotlib.style as style 

style.use('ggplot')



import warnings

warnings.filterwarnings('ignore')

import gc

gc.enable()



!ls ../input/

print ("Ready !")
data = pd.read_csv('../input/X_train.csv')

tr = pd.read_csv('../input/X_train.csv')

sub = pd.read_csv('../input/sample_submission.csv')

test = pd.read_csv('../input/X_test.csv')

target = pd.read_csv('../input/y_train.csv')

print ("Data is ready !!")
data[0:130]
test.head()
target.head()
len(data.measurement_number.value_counts())
data.describe()
test.describe()
target.describe()
totalt = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])

print ("Missing Data at Training")

missing_data.tail()
totalt = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])

print ("Missing Data at Training")

missing_data.tail()
sns.set(style='darkgrid')

sns.countplot(y = 'surface',

              data = target,

              order = target['surface'].value_counts().index)

plt.show()
serie1 = tr.head(128)

serie1.head()
serie1.describe()
plt.figure(figsize=(26, 16))

for i, col in enumerate(serie1.columns[3:]):

    plt.subplot(3, 4, i + 1)

    plt.plot(serie1[col])

    plt.title(col)
series_dict = {}

for series in (data['series_id'].unique()):

    series_dict[series] = data[data['series_id'] == series]  
def plotSeries(series_id):

    style.use('ggplot')

    plt.figure(figsize=(28, 16))

    print(target[target['series_id'] == series_id]['surface'].values[0].title())

    for i, col in enumerate(series_dict[series_id].columns[3:]):

        if col.startswith("o"):

            color = 'red'

        elif col.startswith("a"):

            color = 'green'

        else:

            color = 'blue'

        if i >= 7:

            i+=1

        plt.subplot(3, 4, i + 1)

        plt.plot(series_dict[series_id][col], color=color, linewidth=3)

        plt.title(col)
id_series = 4

plotSeries(id_series)
f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(tr.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(test.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.figure(figsize=(26, 16))

for i, col in enumerate(tr.columns[3:]):

    plt.subplot(3, 4, i + 1)

    plt.hist(tr[col], color='blue', bins=100)

    plt.hist(test[col], color='green', bins=100)

    plt.title(col)
train_df = tr[['series_id']].drop_duplicates().reset_index(drop=True)
for col in tr.columns:

    if 'orient' in col:

        scaler = StandardScaler()

        tr[col] = scaler.fit_transform(tr[col].values.reshape(-1, 1))

        test[col] = scaler.transform(test[col].values.reshape(-1, 1))
from tqdm import tqdm_notebook

from sklearn.linear_model import LinearRegression



def calc_change_rate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)



def add_trend_feature(arr, abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1, 1), arr)

    return lr.coef_[0]



def classic_sta_lta(x, length_sta, length_lta):

    

    sta = np.cumsum(x ** 2)



    # Convert to float

    sta = np.require(sta, dtype=np.float)



    # Copy for LTA

    lta = sta.copy()



    # Compute the STA and the LTA

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]

    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]

    lta /= length_lta



    # Pad zeros

    sta[:length_lta - 1] = 0



    # Avoid division by zero by setting zero values to tiny float

    dtiny = np.finfo(0.0).tiny

    idx = lta < dtiny

    lta[idx] = dtiny



    return sta / lta



for col in tqdm_notebook(tr.columns[3:]):

    train_df[col + '_mean'] = tr.groupby(['series_id'])[col].mean()

    train_df[col + '_std'] = tr.groupby(['series_id'])[col].std()

    train_df[col + '_max'] = tr.groupby(['series_id'])[col].max()

    train_df[col + '_min'] = tr.groupby(['series_id'])[col].min()

    train_df[col + '_max_to_min'] = train_df[col + '_max'] / train_df[col + '_min']



    for i in train_df['series_id']:

        train_df.loc[i, col + '_mean_change_abs'] = np.mean(np.diff(tr.loc[tr['series_id'] == i, col]))

        train_df.loc[i, col + '_mean_change_rate'] = calc_change_rate(tr.loc[tr['series_id'] == i, col])

        

        train_df.loc[i, col + '_q95'] = np.quantile(tr.loc[tr['series_id'] == i, col], 0.95)

        train_df.loc[i, col + '_q99'] = np.quantile(tr.loc[tr['series_id'] == i, col], 0.99)

        train_df.loc[i, col + '_q05'] = np.quantile(tr.loc[tr['series_id'] == i, col], 0.05)

        

        train_df.loc[i, col + '_abs_min'] = np.abs(tr.loc[tr['series_id'] == i, col]).min()

        train_df.loc[i, col + '_abs_max'] = np.abs(tr.loc[tr['series_id'] == i, col]).max()

        

        train_df.loc[i, col + '_trend'] = add_trend_feature(tr.loc[tr['series_id'] == i, col].values)

        train_df.loc[i, col + '_abs_trend'] = add_trend_feature(tr.loc[tr['series_id'] == i, col].values, abs_values=True)

        train_df.loc[i, col + '_abs_mean'] = np.abs(tr.loc[tr['series_id'] == i, col]).mean()

        train_df.loc[i, col + '_abs_std'] = np.abs(tr.loc[tr['series_id'] == i, col]).std()

        

        train_df.loc[i, col + '_mad'] = tr.loc[tr['series_id'] == i, col].mad()

        train_df.loc[i, col + '_kurt'] = tr.loc[tr['series_id'] == i, col].kurtosis()

        train_df.loc[i, col + '_skew'] = tr.loc[tr['series_id'] == i, col].skew()

        train_df.loc[i, col + '_med'] = tr.loc[tr['series_id'] == i, col].median()

        

        train_df.loc[i, col + 'iqr'] = np.subtract(*np.percentile(tr.loc[tr['series_id'] == i, col], [75, 25]))

        train_df.loc[i, col + 'ave10'] = stats.trim_mean(tr.loc[tr['series_id'] == i, col], 0.1)
test_df = sub[['series_id']]

for col in tqdm_notebook(test.columns[3:]):

    test_df[col + '_mean'] = test.groupby(['series_id'])[col].mean()

    test_df[col + '_std'] = test.groupby(['series_id'])[col].std()

    test_df[col + '_max'] = test.groupby(['series_id'])[col].max()

    test_df[col + '_min'] = test.groupby(['series_id'])[col].min()

    test_df[col + '_max_to_min'] = test_df[col + '_max'] / test_df[col + '_min']



    for i in test_df['series_id']:

        test_df.loc[i, col + '_mean_change_abs'] = np.mean(np.diff(test.loc[test['series_id'] == i, col]))

        test_df.loc[i, col + '_mean_change_rate'] = calc_change_rate(test.loc[test['series_id'] == i, col])

        

        test_df.loc[i, col + '_q95'] = np.quantile(test.loc[test['series_id'] == i, col], 0.95)

        test_df.loc[i, col + '_q99'] = np.quantile(test.loc[test['series_id'] == i, col], 0.99)

        test_df.loc[i, col + '_q05'] = np.quantile(test.loc[test['series_id'] == i, col], 0.05)

        

        test_df.loc[i, col + '_abs_min'] = np.abs(test.loc[test['series_id'] == i, col]).min()

        test_df.loc[i, col + '_abs_max'] = np.abs(test.loc[test['series_id'] == i, col]).max()

        

        test_df.loc[i, col + '_trend'] = add_trend_feature(test.loc[test['series_id'] == i, col].values)

        test_df.loc[i, col + '_abs_trend'] = add_trend_feature(test.loc[test['series_id'] == i, col].values, abs_values=True)

        test_df.loc[i, col + '_abs_mean'] = np.abs(test.loc[test['series_id'] == i, col]).mean()

        test_df.loc[i, col + '_abs_std'] = np.abs(test.loc[test['series_id'] == i, col]).std()

        

        test_df.loc[i, col + '_mad'] = test.loc[test['series_id'] == i, col].mad()

        test_df.loc[i, col + '_kurt'] = test.loc[test['series_id'] == i, col].kurtosis()

        test_df.loc[i, col + '_skew'] = test.loc[test['series_id'] == i, col].skew()

        test_df.loc[i, col + '_med'] = test.loc[test['series_id'] == i, col].median()

        

        test_df.loc[i, col + 'iqr'] = np.subtract(*np.percentile(test.loc[test['series_id'] == i, col], [75, 25]))

        test_df.loc[i, col + 'ave10'] = stats.trim_mean(test.loc[test['series_id'] == i, col], 0.1)
train_df.head()
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, train_test_split, GroupKFold, GroupShuffleSplit

n_fold = 2

folds = GroupShuffleSplit(n_splits=n_fold, test_size=0.2, random_state=11)
le = LabelEncoder()

le.fit(target['surface'])

target['surface'] = le.transform(target['surface'])



train_df = train_df.drop(['series_id'], axis=1)

test_df = test_df.drop(['series_id'], axis=1)
import lightgbm as lgb

import xgboost as xgb



def eval_acc(preds, dtrain):

    labels = dtrain.get_label()

    return 'acc', accuracy_score(labels, preds.argmax(1)), True



def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None, groups=target['group_id']):



    oof = np.zeros((len(X), 9))

    prediction = np.zeros((len(X_test), 9))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y, groups)):

        print('Fold', fold_n)

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators = 10000, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='multi_logloss',

                    verbose=5000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict_proba(X_valid)

            score = accuracy_score(y_valid, y_pred_valid.argmax(1))

            print(f'Fold {fold_n}. Accuracy: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_test)

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=20000,  eval_metric='MAE', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid

        scores.append(accuracy_score(y_valid, y_pred_valid.argmax(1)))



        prediction += y_pred    

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof, prediction, feature_importance

        return oof, prediction

    

    else:

        return oof, prediction
params = {'num_leaves': 123,

          'min_data_in_leaf': 12,

          'objective': 'multiclass',

          'max_depth': 22,

          'learning_rate': 0.04680350949723872,

          "boosting": "gbdt",

          "bagging_freq": 5,

          "bagging_fraction": 0.8933018355190274,

          "bagging_seed": 11,

          "verbosity": -1,

          'reg_alpha': 0.9498109326932401,

          'reg_lambda': 0.8058490960546196,

          "num_class": 9,

          'nthread': -1,

          'min_split_gain': 0.009913227240564853,

          'subsample': 0.9027358830703129

         }



#import sklearn.ensemble

#model = RandomForestClassifier(n_estimators=100)

oof_lgb, prediction_lgb, feature_importance = train_model(X=train_df, X_test=test_df, y=target['surface'], params=params, model_type='lgb', model=model, plot_feature_importance=True)
# I use code from this kernel: https://www.kaggle.com/theoviel/deep-learning-starter

import itertools



def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):

    cm = confusion_matrix(truth, pred)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title('Confusion matrix', size=15)

    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.grid(False)

    plt.tight_layout()
plot_confusion_matrix(target['surface'], oof_lgb.argmax(1), le.classes_)