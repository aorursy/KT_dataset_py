%reset -f

import warnings
warnings.filterwarnings('ignore')
import sys
import random
import gc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline

matplotlib.rc('figure', figsize=(14, 11))

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
from plotly import tools
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff

init_notebook_mode(connected=True)
import scipy
from scipy import stats
from scipy.special import boxcox1p
from scipy.special import inv_boxcox1p
import sklearn
import sklearn.decomposition
import sklearn.model_selection
import lightgbm as lgb

from catboost import CatBoostRegressor, Pool
random.seed(321)
np.random.seed(321)
np.set_printoptions(suppress=True)

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
def getContinuousVariableDistributionGraph(dataset_rowcount_tuple, target_value, title = '') :
    figureCVDG = tools.make_subplots(rows=1, cols=2, 
                                     subplot_titles=('Distribution Graph', 'Distribution Graph - Histogram',),
                                    )
    figureCVDG.append_trace(go.Scatter(x=dataset_rowcount_tuple, y=np.sort(target_value), mode='lines', connectgaps=True,),
                            1,1,
                           )
    figureCVDG.append_trace(go.Histogram(x=target_value), 
                            1,2,
                           )
    figureCVDG['layout'].update(title = title, titlefont = dict(family = 'Arial', size = 36,), 
                                paper_bgcolor = '#ffffcf', plot_bgcolor = '#ffffcf',
                               )
    return py.iplot(figureCVDG)


def getCategoricalVariableDistributionGraph(target_value, title = '') :
    tmp_count = target_value.value_counts()
    figureCVDG = tools.make_subplots(rows=1, cols=2, shared_yaxes=True,
                                     subplot_titles=('Distribution Graph', 'Distribution Graph - Bar',),
                                    )
    figureCVDG.append_trace(go.Scatter(x=tmp_count.index, y=tmp_count, mode='markers+lines', connectgaps=True,),
                            1,1,
                           )
    figureCVDG.append_trace(go.Bar(x=tmp_count.index, y=tmp_count), 
                            1,2,
                           )
    figureCVDG['layout'].update(title = title, titlefont = dict(family = 'Arial', size = 36,), 
                                paper_bgcolor = '#ffffcf', plot_bgcolor = '#ffffcf',
                               )
    py.iplot(figureCVDG)
def getDatasetInformation(csv_filepath) :
    """
    Read CSV (comma-separated) file into DataFrame
    
    Returns,
    - DataFrame
    - DataFrame's shape
    - DataFrame's data types
    - DataFrame's describe
    - DataFrame's sorted unique value count
    - DataFrame's missing or NULL value count
    - DataFrame's correlation between numerical columns
    """
    dataset_tmp = pd.read_csv(csv_filepath)
    
    dataset_tmp_shape = pd.DataFrame(list(dataset_tmp.shape), index=['No of Rows', 'No of Columns'], columns=['Total'])
    dataset_tmp_shape = dataset_tmp_shape.reset_index()
    
    dataset_tmp_dtypes = dataset_tmp.dtypes.reset_index()
    dataset_tmp_dtypes.columns = ['Column Names', 'Column Data Types']
    
    dataset_tmp_desc = pd.DataFrame(dataset_tmp.describe())
    dataset_tmp_desc = dataset_tmp_desc.transpose()

    dataset_tmp_unique = dataset_tmp.nunique().reset_index()
    dataset_tmp_unique.columns = ["Column Name", "Unique Value(s) Count"]
    
    dataset_tmp_missing = dataset_tmp.isnull().sum(axis=0).reset_index()
    dataset_tmp_missing.columns = ['Column Names', 'NULL value count per Column']
    dataset_tmp_missing = dataset_tmp_missing.sort_values(by='NULL value count per Column', ascending=False)
    
    # dataset_tmp_corr = dataset_tmp.corr(method='spearman')
    dataset_tmp_corr = pd.DataFrame()
    
    return [dataset_tmp, dataset_tmp_shape, dataset_tmp_dtypes, dataset_tmp_desc, dataset_tmp_unique, dataset_tmp_missing, dataset_tmp_corr]
dataset_santandervp_train, df_shape, df_dtypes, df_describe, df_unique, df_missing, df_corr = getDatasetInformation('../input/train.csv')
dataset_santandervp_train.head()
df_shape
df_dtypes
df_describe
df_unique
df_missing
del(df_shape, df_dtypes, df_describe, df_unique, df_missing, df_corr)
dataset_santandervp_train_target = dataset_santandervp_train['target']
getContinuousVariableDistributionGraph(tuple(range(dataset_santandervp_train.shape[0])), dataset_santandervp_train_target, 'Target - Distribution')
getContinuousVariableDistributionGraph(tuple(range(dataset_santandervp_train.shape[0])), np.log1p(dataset_santandervp_train_target), 'Target - log1p - Distribution')
dataset_santandervp_train_target = \
    np.log1p(dataset_santandervp_train_target)
dataset_santandervp_train.drop(['ID', 'target'], axis=1, inplace=True)
#dataset_santandervp_train.nunique()
santandervp_columns_one_unique_value = \
    dataset_santandervp_train.columns[dataset_santandervp_train.nunique()
        == 1].tolist()
dataset_santandervp_train.drop(santandervp_columns_one_unique_value,
                               axis=1, inplace=True)
dataset_santandervp_train = dataset_santandervp_train.replace([0, 1563411.76],
        np.nan)
dataset_santandervp_train = np.log1p(dataset_santandervp_train)
dataset_santandervp_train.head()
dataset_santandervp_train_tmp = pd.DataFrame()
dataset_santandervp_train_tmp['svp_quantile'] = \
    dataset_santandervp_train[dataset_santandervp_train
                              > 0].quantile(0.5, axis=1)
dataset_santandervp_train_tmp['svp_max'] = \
    dataset_santandervp_train.max(axis=1)
dataset_santandervp_train_tmp['svp_sum_nonzeroval'] = \
    (dataset_santandervp_train != 0).sum(axis=1)
dataset_santandervp_train_tmp['svp_skew'] = \
    dataset_santandervp_train.skew(axis=1)
dataset_santandervp_train_tmp['svp_kurtosis'] = \
    dataset_santandervp_train.kurtosis(axis=1)
dataset_santandervp_train_tmp['svp_sum_all'] = \
    dataset_santandervp_train.sum(axis=1)
dataset_santandervp_train_tmp['svp_variance'] = \
    dataset_santandervp_train.var(axis=1)

dataset_santandervp_train_tmp.head()
dataset_santandervp_train = pd.concat([dataset_santandervp_train,
        dataset_santandervp_train_tmp], axis=1)
dataset_santandervp_train.head(15)
(X_train, X_test, y_train, y_test) = \
    sklearn.model_selection.train_test_split(dataset_santandervp_train,
        dataset_santandervp_train_target, test_size=0.20)
catboost_params = {
    'iterations': 10000,
    'learning_rate': 0.05,
    'depth': 11,
    'l2_leaf_reg': 40,
    'bootstrap_type': 'Bernoulli',
    'eval_metric': 'RMSE',
    'metric_period': 100,
    'od_type': 'Iter',
    'od_wait': 80,
    'allow_writing_files': False,
    'task_type': 'CPU',
    'thread_count': 4,
    }

catboost_model = CatBoostRegressor(**catboost_params)
catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=[], use_best_model=True)
dataset_transformed_train_lgb = lgb.Dataset(data=X_train, label=y_train)
dataset_transformed_test_lgb = lgb.Dataset(data=X_test, label=y_test)
watchlist = [dataset_transformed_train_lgb,
             dataset_transformed_test_lgb]

evaluation_results = {}

lightgbm_params = {
    'objective': 'regression_l2',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'num_leaves': 144,
    'learning_rate': 0.004,
    'bagging_fraction': 0.4,
    'feature_fraction': 0.6,
    'bagging_freq': 4,
    'max_depth': 4,
    'reg_alpha': 1,
    'reg_lambda': 0.1,
    'min_child_weight': 10,
    'zero_as_missing': True,
    'is_training_metric': True,
    'verbosity': -1,
    'device': 'cpu',
    'nthread': 4,
    }

lightgbm_model = lgb.train(
    lightgbm_params,
    train_set=dataset_transformed_train_lgb,
    num_boost_round=10000,
    valid_sets=watchlist,
    early_stopping_rounds=100,
    evals_result=evaluation_results,
    verbose_eval=100,
    )
ax = lgb.plot_metric(evaluation_results, metric='rmse')
plt.show()
ax = lgb.plot_importance(lightgbm_model, max_num_features=52)
plt.show()
max_num_features=37
importance = lightgbm_model.feature_importance()
feature_name = lightgbm_model.feature_name()
tuples = sorted(zip(feature_name, importance), key=lambda x: x[1])
tuples = tuples[-max_num_features:]
labels, values = zip(*tuples)


(X_train, X_test) = (X_train.loc[ : , labels], X_test.loc[ : , labels])
dataset_transformed_train_lgb = lgb.Dataset(data=X_train, label=y_train)
dataset_transformed_test_lgb = lgb.Dataset(data=X_test, label=y_test)
watchlist = [dataset_transformed_train_lgb,
             dataset_transformed_test_lgb]

evaluation_results = {}

lightgbm_params = {
    'objective': 'regression_l2',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'num_leaves': 144,
    'learning_rate': 0.004,
    'bagging_fraction': 0.4,
    'feature_fraction': 0.6,
    'bagging_freq': 4,
    'max_depth': 4,
    'reg_alpha': 1,
    'reg_lambda': 0.1,
    'min_child_weight': 10,
    'zero_as_missing': True,
    'is_training_metric': True,
    'verbosity': -1,
    'device': 'cpu',
    'nthread': 4,
    }

lightgbm_model = lgb.train(
    lightgbm_params,
    train_set=dataset_transformed_train_lgb,
    num_boost_round=10000,
    valid_sets=watchlist,
    early_stopping_rounds=100,
    evals_result=evaluation_results,
    verbose_eval=100,
    )
ax = lgb.plot_metric(evaluation_results, metric='rmse')
plt.show()
del(dataset_santandervp_train, dataset_santandervp_train_target, dataset_santandervp_train_tmp, dataset_transformed_train_lgb, dataset_transformed_test_lgb, watchlist, X_train, X_test, y_train, y_test)
gc.collect()
dataset_santandervp_test, df_shape, df_dtypes, df_describe, df_unique, df_missing, df_corr = getDatasetInformation('../input/test.csv')
dataset_santandervp_test.head()
del(df_shape, df_dtypes, df_describe, df_unique, df_missing, df_corr)
dataset_santandervp_test_ID = dataset_santandervp_test['ID']
dataset_santandervp_test.drop(['ID'], axis=1, inplace=True)
dataset_santandervp_test.drop(santandervp_columns_one_unique_value,
                              axis=1, inplace=True)
dataset_santandervp_test = dataset_santandervp_test.replace([0, 1563411.76],
        np.nan)
dataset_santandervp_test = np.log1p(dataset_santandervp_test)
dataset_santandervp_test.head()
dataset_santandervp_test_tmp = pd.DataFrame()
dataset_santandervp_test_tmp['svp_quantile'] = \
    dataset_santandervp_test[dataset_santandervp_test
                             > 0].quantile(0.5, axis=1)
dataset_santandervp_test_tmp['svp_max'] = \
    dataset_santandervp_test.max(axis=1)
dataset_santandervp_test_tmp['svp_sum_nonzeroval'] = \
    (dataset_santandervp_test != 0).sum(axis=1)
dataset_santandervp_test_tmp['svp_skew'] = \
    dataset_santandervp_test.skew(axis=1)
dataset_santandervp_test_tmp['svp_kurtosis'] = \
    dataset_santandervp_test.kurtosis(axis=1)
dataset_santandervp_test_tmp['svp_sum_all'] = \
    dataset_santandervp_test.sum(axis=1)
dataset_santandervp_test_tmp['svp_variance'] = \
    dataset_santandervp_test.var(axis=1)

dataset_santandervp_test_tmp.head()
dataset_santandervp_test = pd.concat([dataset_santandervp_test,
        dataset_santandervp_test_tmp], axis=1)
dataset_santandervp_test.head(15)
catboost_model_predictions_test = \
    catboost_model.predict(dataset_santandervp_test)
catboost_model_predictions_test
catboost_model_predictions_test = \
    np.expm1(catboost_model_predictions_test)
catboost_model_predictions_test
dataset_santandervp_test = dataset_santandervp_test.loc[ : , labels]
lightgbm_model_predictions_test = \
    lightgbm_model.predict(dataset_santandervp_test,
                           num_iteration=lightgbm_model.best_iteration)
lightgbm_model_predictions_test
lightgbm_model_predictions_test = \
    np.expm1(lightgbm_model_predictions_test)
lightgbm_model_predictions_test
dataset_submission = pd.DataFrame()
dataset_submission['ID'] = dataset_santandervp_test_ID
dataset_submission['target'] = ((lightgbm_model_predictions_test * 0.5) + (catboost_model_predictions_test * 0.5))
dataset_submission.to_csv('submission.csv', index=False,
                          float_format='%.2f')
dataset_submission.head()
dataset_submission_lightgbm = pd.DataFrame()
dataset_submission_lightgbm['ID'] = dataset_santandervp_test_ID
dataset_submission_lightgbm['target'] = ((lightgbm_model_predictions_test))
dataset_submission_lightgbm.to_csv('submission_lightgbm.csv', index=False,
                          float_format='%.2f')
dataset_submission_lightgbm.head()
dataset_submission_catboost = pd.DataFrame()
dataset_submission_catboost['ID'] = dataset_santandervp_test_ID
dataset_submission_catboost['target'] = ((catboost_model_predictions_test))
dataset_submission_catboost.to_csv('submission_catboost.csv', index=False,
                          float_format='%.2f')
dataset_submission_catboost.head()