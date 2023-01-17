import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import os

import gc

from tqdm.notebook import tqdm

from sklearn.metrics import f1_score

from tqdm.notebook import tqdm

from pykalman import KalmanFilter

from scipy.misc import derivative

from bayes_opt import BayesianOptimization



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor



import torch

import torch.nn as nn

from torch.autograd import Variable

from torch import optim

from torch.optim.lr_scheduler import OneCycleLR

from torch.nn import functional as f

from torch.utils.data import Dataset, DataLoader

import warnings



sns.set_style("darkgrid")

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



base_path = '../input/clean-kalman/'

base_path0 = '../input/liverpool-ion-switching'

train_path = os.path.join(base_path, 'train_clean_kalman.csv')

test_path = os.path.join(base_path, 'test_clean_kalman.csv')

sub_path = os.path.join(base_path0, 'sample_submission.csv')



fold_split = 10

window_list = [20, 50]

SAMPLE = 10000

FOLD_GET = 5

RANDOM_NOISE = 0.001



kf = StratifiedKFold(n_splits=fold_split, random_state=37, shuffle=True)
train_df = pd.read_csv(train_path)

sns.distplot(train_df['signal'].values)

plt.show()

train_df.tail()
train_df['signal'] = train_df['signal'].apply(lambda x: x * np.random.uniform(1-RANDOM_NOISE, 1+RANDOM_NOISE))

sns.distplot(train_df['signal'].values)

plt.show()

train_df.tail()
test_df = pd.read_csv(test_path)

sns.distplot(test_df['signal'].values)

plt.show()

test_df.tail()
%%time



def Kalman1D(observations,damping=1):

    # To return the smoothed time series data

    observation_covariance = damping

    initial_value_guess = observations[0]

    transition_matrix = 1

    transition_covariance = 0.1

    initial_value_guess

    kf = KalmanFilter(

            initial_state_mean=initial_value_guess,

            initial_state_covariance=observation_covariance,

            observation_covariance=observation_covariance,

            transition_covariance=transition_covariance,

            transition_matrices=transition_matrix

        )

    pred_state, state_cov = kf.smooth(observations)

    return pred_state



observation_covariance = 0.0015



# train_df['signal'] = Kalman1D(train_df['signal'].values,observation_covariance)

# test_df['signal'] = Kalman1D(test_df['signal'].values,observation_covariance)
def normalize(train, test, col, type_='StandardScaler'):

    

    if type_ == 'StandardScaler':

        clf = StandardScaler()

    elif type_ == 'MinMaxScaler':

        clf = MinMaxScaler()

    elif type_ == 'RobustScaler':

        clf = RobustScaler()

    else:

        print('Error type in!')

    

    train[col] = clf.fit_transform(train[col])

    test[col] = clf.transform(test[col])

    

    return train, test



train_df, test_df = normalize(train_df, test_df, ['signal'])

gc.collect()
channel_signal = train_df.groupby('open_channels')['signal'].agg(['min', 'max', np.mean, np.std]).reset_index()

channel_signal.columns = ['signal_' + i if i!='open_channels' else i for i in channel_signal.columns]



print(channel_signal)



try:

    del channel_signal

except:

    print('Variable channel_signal not defined!')
def process_data(df, windows=window_list, batch_size=20000):    

    

    df = df.sort_values('time').reset_index(drop=True)

    df.index = ((df.time*10000) - 1).values

    df['batch'] = df.index // batch_size

    df['batch_index'] = df.index % batch_size

    df['batch_slices'] = df['batch_index'] // (batch_size//10)

    df['batch_slices1'] = df.apply(lambda x: '_'.join([str(x['batch']).zfill(3), str(x['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch', 'batch_slices1']:

        d = {}

        

        d[f'min_{c}'] = df.groupby(c)['signal'].min()

        d[f'max_{c}'] = df.groupby(c)['signal'].max()

        d[f'mean_{c}'] = df.groupby(c)['signal'].mean()

        d[f'std_{c}'] = df.groupby(c)['signal'].std()

#         d[f'median_{c}'] = df.groupby(c)['signal'].median()

        

        for per in [10, 25, 75, 90]:

            d[f'pct_{per}_{c}'] = df.groupby(c)['signal'].apply(lambda x: np.percentile(x, per))

        

        d[f'skew_{c}'] = df.groupby(c)['signal'].apply(lambda x: pd.Series(x).skew())

        d[f'kurtosis_{c}'] = df.groupby(c)['signal'].apply(lambda x: pd.Series(x).kurtosis())

        d[f'mean_abs_{c}'] = df.groupby(c)['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d[f'min_abs_{c}'] =  df.groupby(c)['signal'].apply(lambda x: np.min(np.abs(x)))

        d[f'max_abs_{c}'] =  df.groupby(c)['signal'].apply(lambda x: np.max(np.abs(x)))

        d[f'range_{c}'] = d[f'max_{c}'] - d[f'min_{c}']

        d[f'ratio_{c}'] = d[f'max_{c}'] / d[f'min_{c}']

        d[f'avg_abs_{c}'] = (d[f'min_abs_{c}'] + d[f'max_abs_{c}'])/2

        

        for v in d:

            df[v] = df[c].map(d[v].to_dict())

    

    # add shift_1

    df['shift_+1'] = [0] + list(df['signal'].values[:-1])

    df['shift_-1'] = list(df['signal'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['shift_+1'][i] = np.nan

    for i in df[df['batch_index']==(batch_size-1)].index:

        df['shift_-1'][i] = np.nan

    

    # add shift_2

    df['shift_+2'] = [0,] + [1,] + list(df['signal'].values[:-2])

    df['shift_-2'] = list(df['signal'].values[:-2]) + [0,] + [1,]

    

    for i in df[df['batch_index'].isin([0, 1])].index:

        df['shift_+2'][i] = np.nan

    for i in df[df['batch_index'].isin([batch_size-1, batch_size-2])].index:

        df['shift_-2'][i] = np.nan

        

    df = df.drop(columns=['batch', 'batch_index', 'batch_slices', 'batch_slices1'])

    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels']]:

        df[f'{c}_msignal'] = df[f'{c}'] - df['signal']        

        

    df = df.replace([np.Inf, -np.Inf], np.nan)

    df.fillna(0, inplace=True)

    

    gc.collect()

    

    return df



train_df = process_data(train_df)

test_df = process_data(test_df)
def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype

    if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:

                    df[col] = df[col].astype(np.uint8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:

                    df[col] = df[col].astype(np.uint16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:

                    df[col] = df[col].astype(np.uint32)                    

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:

                    df[col] = df[col].astype(np.uint64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
%%time



train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)

gc.collect()
def feature_important(importances, title):

    

    data = pd.DataFrame({'feature': column_train, 'important': importances})

    

    plt.figure(figsize=(25, 25))

    plt.title('Feature Importances')

    sns.barplot(data=data.sort_values('important', ascending=False), x='important', y='feature')

    plt.xlabel('Relative Importance')

    plt.title(title)

    plt.show()
column_train = [i for i in train_df.columns if i not in ['time', 'signal', 'open_channels']]

column_train
def custom_asymmetric_train(y_pred, y_true):

    

    y_true = y_true.get_label()

    residual = (y_true - y_pred).astype('float')

    grad = np.where(residual < 0, -2*1.1*residual, -2*residual)

    hess = np.where(residual < 0, 2*1.1, 2)

    

    return grad, hess







def custom_asymmetric_valid(y_pred, y_true):

    

    y_true = y_true.get_label()

    residual = (y_true - y_pred).astype('float')

    loss = np.where(residual < 0, residual**2, 1.15*(residual**2))

    

    return 'custom_asymmetric_eval', np.mean(loss), False









def eval_gini(y_true, y_prob):

    return np.sqrt(((y_prob - y_true) ** 2).mean())
def model_train(model_type, params, train_df=train_df, test_df=test_df):    

        

    predict = np.zeros([len(test_df)])

    importances = np.zeros([len(column_train), FOLD_GET])



    for idx, (train_index, val_index) in enumerate(kf.split(train_df, train_df['open_channels'])):

        if idx % 2 == 0:

            print(f'Fold_{idx}:')

            train_ = train_df.iloc[train_index]

            val_ = train_df.iloc[val_index]



            train_.reset_index(inplace=True, drop=True)

            val_.reset_index(inplace=True, drop=True)            



            if model_type=='xgb':

                

                train_dataset = lgb.DMatrix(train_.loc[:, column_train].values, label=train_.loc[:, 'open_channels'].values, feature_name=column_train)    

                val_dataset = lgb.DMatrix(val_.loc[:, column_train].values, label=val_.loc[:, 'open_channels'].values, feature_name=column_train)

                

                clf = xgb.train(params, train_dataset, 500, valid_sets=[(val_dataset, 'valid')],

                          verbose_eval=100, early_stopping_rounds = 100)

                predict += clf.predict(test_df.loc[:, column_train], ntree_limit=clf.best_ntree_limit)/FOLD_GET

                lgb.plot_importance(clf, max_num_features=10)

                



            elif model_type=='lgb':   

                

                train_dataset = lgb.Dataset(train_.loc[:, column_train].values, label=train_.loc[:, 'open_channels'].values, feature_name=column_train)    

                val_dataset = lgb.Dataset(val_.loc[:, column_train].values, label=val_.loc[:, 'open_channels'].values, feature_name=column_train)

            

                clf = lgb.train(params, train_dataset, 4000, valid_sets=[train_dataset, val_dataset], feval=custom_asymmetric_valid,

                                fobj=custom_asymmetric_train, verbose_eval=500, early_stopping_rounds = 400)

                predict += clf.predict(test_df.loc[:, column_train], ntree_limit=clf.best_iteration)/FOLD_GET            



            del clf

            gc.collect()



    return predict, importances
# train_sample = train_df.sample(SAMPLE, replace=True)

# train_sample.reset_index(inplace=True, drop=True)



train_index, val_index = list(kf.split(train_df, train_df['open_channels']))[0]

train_ = train_df.iloc[train_index]

val_ = train_df.iloc[val_index]



train_.reset_index(inplace=True, drop=True)

val_.reset_index(inplace=True, drop=True)



# del train_sample

gc.collect()
# %%time



# def xgb_bayesian(    

#     max_depth,

#     scale_pos_weight,

#     gamma,

#     eta,

#     subsample,

#     colsample_bytree,

#     min_child_weight,

#     max_delta_step

# ):

    

#     max_depth = int(max_depth)

#     scale_pos_weight = int(scale_pos_weight)

#     min_child_weight = int(min_child_weight)



#     assert type(max_depth) == int

#     assert type(scale_pos_weight) == int

#     assert type(min_child_weight) == int



#     param = {

#         'max_depth': max_depth,

#         'gamma': gamma,

#         'eta': eta,

#         'objective': 'reg:squarederror',

#         'nthread': 4,

#         'eval_metric': 'rmse',

#         'subsample': subsample,

#         'colsample_bytree': colsample_bytree,

#         'min_child_weight': min_child_weight,

#         'max_delta_step': max_delta_step

#     }    

    

    

#     xgb_train = xgb.DMatrix(train_.loc[:, column_train].values, label=train_.loc[:, 'open_channels'].values)    

#     xgb_valid = xgb.DMatrix(val_.loc[:, column_train].values, label=val_.loc[:, 'open_channels'].values)   



#     clf = xgb.train(param, dtrain=xgb_train, num_boost_round=10, evals=[(xgb_valid, 'valid')],

#                     verbose_eval=10, early_stopping_rounds = 400)



#     predictions = clf.predict(val_.loc[:, column_train].values)[0]       

#     score = eval_gini(val_.loc[:, "open_channels"].values, predictions)

    

#     return 1-score





# bounds_xgb = {

#     'gamma': (0.001, 10),

#     'eta': (0.01, 0.2),

#     'max_depth':(3,15),

#     'subsample': (0, 0.5),

#     'colsample_bytree': (0, 0.5),

#     'min_child_weight': (0, 20),

#     'max_delta_step': (0, 10),

#     'scale_pos_weight': (1, 10)

# }





# xgb_bo = BayesianOptimization(xgb_bayesian, bounds_xgb, random_state=37)

# print(xgb_bo.space.keys)



# xgb_bo.maximize(init_points=5, n_iter=5, acq='ucb', xi=0.0, alpha=1e-6)

# xgb_bo.max['params']
# base_param = {

#         'max_depth': int(xgb_bo.max['params']['max_depth'])

#         'gamma': xgb_bo.max['params']['gamma'],

#         'eta': xgb_bo.max['params']['eta'],

#         'objective': 'reg:squarederror',

#         'nthread': 4,

#         'eval_metric': 'rmse',

#         'subsample': xgb_bo.max['params']['subsample'],

#         'colsample_bytree': xgb_bo.max['params']['colsample_bytree'],

#         'min_child_weight': int(xgb_bo.max['params']['min_child_weight']),

#         'max_delta_step': int(xgb_bo.max['params']['max_depth'])

#     }
# %%time



# pred_xgb, importances_xgb = model_train('xgb')
%%time



def lgb_bayesian(

    num_leaves,

    min_data_in_leaf,

    learning_rate,

    min_sum_hessian_in_leaf,

    feature_fraction,

    lambda_l1,

    lambda_l2,

    min_gain_to_split,

    max_depth,

    scale_pos_weight):

    

    num_leaves = int(num_leaves)

    min_data_in_leaf = int(min_data_in_leaf)

    max_depth = int(max_depth)

    scale_pos_weight = int(scale_pos_weight)



    assert type(num_leaves) == int

    assert type(min_data_in_leaf) == int

    assert type(max_depth) == int

    assert type(scale_pos_weight) == int



    param = {

        'num_leaves': num_leaves,

        'max_bin': 63,

        'min_data_in_leaf': min_data_in_leaf,

        'learning_rate': learning_rate,

        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,

        'bagging_fraction': 1.0,

        'bagging_freq': 5,

        'feature_fraction': feature_fraction,

        'lambda_l1': lambda_l1,

        'lambda_l2': lambda_l2,

        'min_gain_to_split': min_gain_to_split,

        'max_depth': max_depth,

        'scale_pos_weight': scale_pos_weight,

        'save_binary': True, 

        'objective': 'regression',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'mae',

        'boost_from_average': False,

    }    

    

    

    lgb_train = lgb.Dataset(train_.loc[:, column_train].values, label=train_.loc[:, 'open_channels'].values)    

    lgb_valid = lgb.Dataset(val_.loc[:, column_train].values, label=val_.loc[:, 'open_channels'].values)   



    clf = lgb.train(param, lgb_train, 1000, valid_sets=[lgb_valid],feval=custom_asymmetric_valid,

                    fobj=custom_asymmetric_train, verbose_eval=250, early_stopping_rounds = 400)

    

    predictions = clf.predict(val_.loc[:, column_train].values, num_iteration=clf.best_iteration)       

    score = eval_gini(val_.loc[:, "open_channels"].values, predictions)

    

    return 1-score





bounds_lgb = {

    'num_leaves': (5, 20), 

    'min_data_in_leaf': (5, 20),  

    'learning_rate': (0.01, 0.2),

    'min_sum_hessian_in_leaf': (0.00001, 0.01),    

    'feature_fraction': (0.05, 0.5),

    'lambda_l1': (0, 5.0), 

    'lambda_l2': (0, 5.0), 

    'min_gain_to_split': (0, 1.0),

    'max_depth':(3,15),

    'scale_pos_weight': (1, 10)

}





lgb_bo = BayesianOptimization(lgb_bayesian, bounds_lgb, random_state=37)

print(lgb_bo.space.keys)



lgb_bo.maximize(init_points=5, n_iter=5, acq='ucb', xi=0.0, alpha=1e-6)

lgb_bo.max['params']
base_param = {

        'num_leaves': int(lgb_bo.max['params']['num_leaves']),

        'max_bin': 63,

        'min_data_in_leaf': int(lgb_bo.max['params']['min_data_in_leaf']),

        'learning_rate': lgb_bo.max['params']['learning_rate'],

        'min_sum_hessian_in_leaf': lgb_bo.max['params']['min_sum_hessian_in_leaf'],

        'bagging_fraction': 1.0,

        'bagging_freq': 5,

        'feature_fraction': lgb_bo.max['params']['feature_fraction'],

        'lambda_l1': lgb_bo.max['params']['lambda_l1'],

        'lambda_l2': lgb_bo.max['params']['lambda_l2'],

        'min_gain_to_split': lgb_bo.max['params']['min_gain_to_split'],

        'max_depth': int(lgb_bo.max['params']['max_depth']),

        'scale_pos_weight': int(lgb_bo.max['params']['scale_pos_weight']),

        'save_binary': True, 

        'objective': 'regression',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'mae',

        'boost_from_average': False,

    }
%%time



pred_lgb, importances_lgb = model_train('lgb', base_param)
del train_df, test_df

gc.collect()
def pred_proc(pred):

    pred = np.round(np.clip(pred, 0, 10))

    return pred.astype(int)



# pred = 0.5*pred_xgb + 0.5*pred_lgb

pred = pred_proc(pred_lgb)

set(pred)
submission = pd.read_csv(sub_path)



submission['open_channels'] = pred

submission.to_csv("submission.csv", index=False, float_format='%.4f')



submission.tail()