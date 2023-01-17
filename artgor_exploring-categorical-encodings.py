!pip install -U vega_datasets notebook vega
import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit

from sklearn import metrics

from sklearn import linear_model

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from bayes_opt import BayesianOptimization

import eli5

import shap

from IPython.display import HTML

import json

import altair as alt



import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline



from category_encoders.ordinal import OrdinalEncoder

from category_encoders.woe import WOEEncoder

from category_encoders.target_encoder import TargetEncoder

from category_encoders.sum_coding import SumEncoder

from category_encoders.m_estimate import MEstimateEncoder

from category_encoders.backward_difference import BackwardDifferenceEncoder

from category_encoders.leave_one_out import LeaveOneOutEncoder

from category_encoders.helmert import HelmertEncoder

from category_encoders.cat_boost import CatBoostEncoder

from category_encoders.james_stein import JamesSteinEncoder

from category_encoders.one_hot import OneHotEncoder

from typing import List



import os

import time

import datetime

import json

import gc

from numba import jit



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics



from itertools import product



import altair as alt

from altair.vega import v5

from IPython.display import HTML



alt.renderers.enable('notebook')



%env JOBLIB_TEMP_FOLDER=/tmp
import category_encoders

category_encoders.__version__
# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey

def prepare_altair():

    """

    Helper function to prepare altair for working.

    """



    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION

    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

    noext = "?noext"

    

    paths = {

        'vega': vega_url + noext,

        'vega-lib': vega_lib_url + noext,

        'vega-lite': vega_lite_url + noext,

        'vega-embed': vega_embed_url + noext

    }

    

    workaround = f"""    requirejs.config({{

        baseUrl: 'https://cdn.jsdelivr.net/npm/',

        paths: {paths}

    }});

    """

    

    return workaround

    



def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

           



@add_autoincrement

def render(chart, id="vega-chart"):

    """

    Helper function to plot altair visualizations.

    """

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )

    



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=True).sum() / 1024**2

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

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

    





def prepare_plot_dict(df, col):

    """

    Preparing dictionary with data for plotting.

    

    I want to show how much higher/lower are the target rates for the current column comparing to base values (as described higher),

    At first I calculate base rates, then for each category in the column I calculate target rate and find difference with the base rates.

    

    """

    main_count = train['target'].value_counts(normalize=True).sort_index()

    main_count = dict(main_count)

    plot_dict = {}

    for i in df[col].unique():

        val_count = dict(df.loc[df[col] == i, 'target'].value_counts().sort_index())



        for k, v in main_count.items():

            if k in val_count:

                plot_dict[val_count[k]] = ((val_count[k] / sum(val_count.values())) / main_count[k]) * 100 - 100

            else:

                plot_dict[0] = 0



    return plot_dict



def make_count_plot(df, x, hue='target', title=''):

    """

    Plotting countplot with correct annotations.

    """

    g = sns.countplot(x=x, data=df, hue=hue);

    plt.title(f'Target {title}');

    ax = g.axes



    plot_dict = prepare_plot_dict(df, x)



    for p in ax.patches:

        h = p.get_height() if str(p.get_height()) != 'nan' else 0

        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"

        ax.annotate(text, (p.get_x() + p.get_width() / 2., h),

             ha='center', va='center', fontsize=11, color='green' if plot_dict[h] > 0 else 'red', rotation=0, xytext=(0, 10),

             textcoords='offset points')  



def plot_two_graphs(col: str = '', top_n: int = None):

    """

    Plotting four graphs:

    - target by variable;

    - counts of categories in the variable in train and test;

    """

    data = train.copy()

    all_data1 = all_data.copy()

    if top_n:

        top_cats = list(train[col].value_counts()[:5].index)

        data = data.loc[data[col].isin(top_cats)]

        all_data1 = all_data1.loc[all_data1[col].isin(top_cats)]

        

    plt.figure(figsize=(20, 12));

    plt.subplot(2, 2, 1)

    make_count_plot(df=data, x=col, title=f'and {col}')



    plt.subplot(2, 2, 2)

    sns.countplot(x='dataset_type', data=all_data1, hue=col);

    plt.title(f'Count of samples in {col} in train and test data');



    

@jit

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





class DoubleValidationEncoderNumerical:

    """

    Encoder with validation within

    """

    def __init__(self, cols: List, encoder, folds):

        """

        :param cols: Categorical columns

        :param encoder: Encoder class

        :param folds: Folds to split the data

        """

        self.cols = cols

        self.encoder = encoder

        self.encoders_dict = {}

        self.folds = folds



    def fit_transform(self, X: pd.DataFrame, y: np.array) -> pd.DataFrame:

        X = X.reset_index(drop=True)

        y = y.reset_index(drop=True)

        for n_fold, (train_idx, val_idx) in enumerate(self.folds.split(X, y)):

            X_train, X_val = X.loc[train_idx].reset_index(drop=True), X.loc[val_idx].reset_index(drop=True)

            y_train, y_val = y[train_idx], y[val_idx]

            _ = self.encoder.fit_transform(X_train, y_train)



            # transform validation part and get all necessary cols

            val_t = self.encoder.transform(X_val)



            if n_fold == 0:

                cols_representation = np.zeros((X.shape[0], val_t.shape[1]))

            

            self.encoders_dict[n_fold] = self.encoder



            cols_representation[val_idx, :] += val_t.values



        cols_representation = pd.DataFrame(cols_representation, columns=X.columns)



        return cols_representation



    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.reset_index(drop=True)



        cols_representation = None



        for encoder in self.encoders_dict.values():

            test_tr = encoder.transform(X)



            if cols_representation is None:

                cols_representation = np.zeros(test_tr.shape)



            cols_representation = cols_representation + test_tr / self.folds.n_splits



        cols_representation = pd.DataFrame(cols_representation, columns=X.columns)

        

        return cols_representation





class FrequencyEncoder:

    def __init__(self, cols):

        self.cols = cols

        self.counts_dict = None



    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        counts_dict = {}

        for col in self.cols:

            values, counts = np.unique(X[col], return_counts=True)

            counts_dict[col] = dict(zip(values, counts))

        self.counts_dict = counts_dict



    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        counts_dict_test = {}

        res = []

        for col in self.cols:

            values, counts = np.unique(X[col], return_counts=True)

            counts_dict_test[col] = dict(zip(values, counts))



            # if value is in "train" keys - replace "test" counts with "train" counts

            for k in [key for key in counts_dict_test[col].keys() if key in self.counts_dict[col].keys()]:

                counts_dict_test[col][k] = self.counts_dict[col][k]



            res.append(X[col].map(counts_dict_test[col]).values.reshape(-1, 1))

        res = np.hstack(res)



        X[self.cols] = res

        return X



    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        self.fit(X, y)

        X = self.transform(X)

        return X

    



def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3, averaging='usual', n_jobs=-1, encoder=None, enc_val='single'):

    """

    A function to train a variety of classification models.

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

    columns = X.columns if columns is None else columns

    n_splits = folds.n_splits if splits is None else n_folds

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'AUC',

                        'sklearn_scoring_function': metrics.roc_auc_score},

                    }

    

    result_dict = {}

    if averaging == 'usual':

        # out-of-fold predictions on train data

        oof = np.zeros((len(X), 1))



        # averaged predictions on train data

        prediction = np.zeros((len(X_test), 1))

        

    elif averaging == 'rank':

        # out-of-fold predictions on train data

        oof = np.zeros((len(X), 1))



        # averaged predictions on train data

        prediction = np.zeros((len(X_test), 1))



    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

        

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        if verbose:

            print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        

        X_t = X_test.copy()

            

        if encoder and enc_val == 'single':

            X_train = encoder.fit_transform(X_train, y_train)

            X_valid = encoder.transform(X_valid)

            X_t = encoder.transform(X_t)

        elif encoder and enc_val == 'double':

            encoder_double = DoubleValidationEncoderNumerical(cols=columns, encoder=encoder, folds=folds)

            X_train = encoder_double.fit_transform(X_train, y_train)

            X_valid = encoder_double.transform(X_valid)

            X_t = encoder_double.transform(X_t)

            

            

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = n_jobs)

            model.fit(X_train, y_train, 

                    eval_set=[(X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict_proba(X_valid)[:, 1]

            y_pred = model.predict_proba(X_t, num_iteration=model.best_iteration_)[:, 1]

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_t, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_t)[:, 1]

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function='Logloss')

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_t)

        

        if averaging == 'usual':

            

            oof[valid_index] = y_pred_valid.reshape(-1, 1)

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

            

            prediction += y_pred.reshape(-1, 1)



        elif averaging == 'rank':

                                  

            oof[valid_index] = y_pred_valid.reshape(-1, 1)

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

                                  

            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)        

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_splits

    if verbose:

        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_splits

        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

            by="importance", ascending=False)[:50].index



        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



        result_dict['feature_importance'] = feature_importance

        result_dict['top_columns'] = list(cols)

        if plot_feature_importance:



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

        

    return result_dict



# setting up altair

workaround = prepare_altair()

HTML("".join((

    "<script>",

    workaround,

    "</script>",

)))
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

sample_submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')



train['dataset_type'] = 'train'

test['dataset_type'] = 'test'

all_data = pd.concat([train, test])
train.head()
train.isnull().any().any(), test.isnull().any().any()
for col in train.columns[1:-2]:

    print(col, train[col].nunique())
set(train['nom_7'].unique()) - set(test['nom_7'].unique())
set(test['nom_8'].unique()) - set(train['nom_8'].unique()), set(train['nom_8'].unique()) - set(test['nom_8'].unique())
len(set(test['nom_9'].unique()) - set(train['nom_9'].unique())), len(set(train['nom_9'].unique()) - set(test['nom_9'].unique()))
train['target'].value_counts()
for col in [col for col in train.columns if 'bin' in col]:

    plot_two_graphs(col=col)
for col in [col for col in train.columns if 'nom' in col]:

    plot_two_graphs(col=col, top_n=5)
for col in [col for col in train.columns if 'ord' in col]:

    plot_two_graphs(col=col, top_n=5)
plot_two_graphs(col='day')
plot_two_graphs(col='month')
train['ord_5_1'] = train['ord_5'].apply(lambda x: x[0])

train['ord_5_2'] = train['ord_5'].apply(lambda x: x[1])

test['ord_5_1'] = test['ord_5'].apply(lambda x: x[0])

test['ord_5_2'] = test['ord_5'].apply(lambda x: x[1])
cat_columns = [col for col in train.columns if col not in ['id', 'target', 'dataset_type']]

for col in tqdm_notebook(cat_columns):

    le = LabelEncoder()

    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

    train[col] = le.transform(list(train[col].astype(str).values))

    test[col] = le.transform(list(test[col].astype(str).values))   
X = train.drop(['id', 'target', 'dataset_type'], axis=1)

y = train['target']

X_test = test.drop(['id', 'dataset_type'], axis=1)

del all_data
n_fold = 5

folds = StratifiedKFold(n_splits=5)
params = {'num_leaves': 256,

          'min_child_samples': 79,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.03,

          "boosting_type": "gbdt",

          "subsample_freq": 3,

          "subsample": 1.0,

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3,

          'reg_lambda': 0.3,

          'colsample_bytree': 1.0

         }

result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,

                                                      verbose=500, early_stopping_rounds=200, n_estimators=100, averaging='usual', n_jobs=-1)
encoders_dict = {}

encoders = [TargetEncoder(cols=cat_columns), TargetEncoder(cols=cat_columns, min_samples_leaf=5), TargetEncoder(cols=cat_columns, smoothing=5),

            WOEEncoder(cols=cat_columns), WOEEncoder(cols=cat_columns, sigma=0.01), WOEEncoder(cols=cat_columns, sigma=0.5), WOEEncoder(cols=cat_columns, regularization=0.1), WOEEncoder(cols=cat_columns, regularization=5),

           CatBoostEncoder(cols=cat_columns), CatBoostEncoder(cols=cat_columns, sigma=0.01), CatBoostEncoder(cols=cat_columns, sigma=0.5)]

for i, enc in tqdm_notebook(enumerate(encoders)):

    encoder = enc

    result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=False,

                                                      verbose=False, early_stopping_rounds=200, n_estimators=100, averaging='usual', n_jobs=-1,

                                             encoder=encoder, enc_val='single')

    encoders_dict[i] = [encoder, result_dict_lgb['scores']]
encoders_df = pd.DataFrame(encoders_dict.values(), columns=['encoder', 'scores'])

encoders_df['mean'] = encoders_df['scores'].apply(lambda x: np.mean(x))

for i, row in encoders_df.sort_values('mean', ascending=False)[:5].iterrows():

    print(row['encoder'])

    print(f"Mean score: {row['scores']}")

    print()
encoders_dict_double = {}

for i, enc in tqdm_notebook(enumerate(encoders)):

    encoder = enc

    result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=False,

                                                      verbose=False, early_stopping_rounds=200, n_estimators=100, averaging='usual', n_jobs=-1,

                                             encoder=encoder, enc_val='double')

    encoders_dict_double[i] = [encoder, result_dict_lgb['scores']]
encoders_df = pd.DataFrame(encoders_dict_double.values(), columns=['encoder', 'scores'])

encoders_df['mean'] = encoders_df['scores'].apply(lambda x: np.mean(x))

for i, row in encoders_df.sort_values('mean', ascending=False)[:5].iterrows():

    print(row['encoder'])

    print(f"Mean score: {row['mean']}")

    print()
encoder = encoders_df.sort_values('mean', ascending=False)['encoder'].values[0]

encoder
result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,

                                                      verbose=500, early_stopping_rounds=200, n_estimators=5000, averaging='usual', n_jobs=-1,

                                             encoder=encoder, enc_val='double')
sample_submission['target'] = result_dict_lgb['prediction']

sample_submission.to_csv('submission.csv', index=False)
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')



train['ord_5_1'] = train['ord_5'].apply(lambda x: x[0])

train['ord_5_2'] = train['ord_5'].apply(lambda x: x[1])

test['ord_5_1'] = test['ord_5'].apply(lambda x: x[0])

test['ord_5_2'] = test['ord_5'].apply(lambda x: x[1])
top_columns = result_dict_lgb['top_columns'][:5]

new_cat_columns = []

new_cat_columns.extend(cat_columns)

for col1 in tqdm_notebook(top_columns):

    for col2 in cat_columns:

        if col1 != col2:

            train[f'{col1}_{col2}'] = train[col1].astype(str) + '_' + train[col2].astype(str)

            test[f'{col1}_{col2}'] = test[col1].astype(str) + '_' + test[col2].astype(str)

            new_cat_columns.append(f'{col1}_{col2}')
def process_data(df: pd.DataFrame):

    data = df.copy()

    # additional features based on ord_5

    data['ord_5_1'] = data['ord_5'].apply(lambda x: x[0])

    data['ord_5_2'] = data['ord_5'].apply(lambda x: x[1])

    

    # https://www.kaggle.com/gogo827jz/catboost-baseline-with-feature-importance

    mapper_ord_1 = {'Novice': 1, 

                'Contributor': 2,

                'Expert': 3, 

                'Master': 4, 

                'Grandmaster': 5}



    mapper_ord_2 = {'Freezing': 1, 

                    'Cold': 2, 

                    'Warm': 3, 

                    'Hot': 4,

                    'Boiling Hot': 5, 

                    'Lava Hot': 6}



    mapper_ord_3 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 

                    'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15}



    mapper_ord_4 = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 

                    'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15,

                    'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 

                    'W': 23, 'X': 24, 'Y': 25, 'Z': 26}

    

    for col, mapper in zip(['ord_1', 'ord_2', 'ord_3', 'ord_4'], [mapper_ord_1, mapper_ord_2, mapper_ord_3, mapper_ord_4]):

        data[col] = data[col].replace(mapper)

        

    ord_5 = sorted(list(set(data['ord_5'].values)))

    ord_5 = dict(zip(ord_5, range(len(ord_5))))

    data.loc[:, 'ord_5'] = data['ord_5'].apply(lambda x: ord_5[x]).astype(float)

    



    data['bin_3'] = data['bin_3'].apply(lambda x: 1 if x == 'T' else 0)

    data['bin_4'] = data['bin_4'].apply(lambda x: 1 if x == 'Y' else 0)

    

    # https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning



    def date_cyc_enc(df, col, max_vals):

        df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

        df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

        return df



    data = date_cyc_enc(data, 'day', 7)

    data = date_cyc_enc(data, 'month', 12)

    

    return data



train = process_data(train)

test = process_data(test)
for col in tqdm_notebook(new_cat_columns):

    le = LabelEncoder()

    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

    train[col] = le.transform(list(train[col].astype(str).values))

    test[col] = le.transform(list(test[col].astype(str).values))   
X = train.drop(['id', 'target'], axis=1)

y = train['target']

X_test = test.drop(['id'], axis=1)
result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,

                                                      verbose=100, early_stopping_rounds=200, n_estimators=5000, averaging='usual', n_jobs=-1,

                                             encoder=CatBoostEncoder(cols=new_cat_columns, sigma=0.01), enc_val='double')

sample_submission['target'] = result_dict_lgb['prediction']

sample_submission.to_csv('submission.csv', index=False)