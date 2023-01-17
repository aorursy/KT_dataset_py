# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.set_config_file(offline=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_2037160701 = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/2037160701.csv')

df_2037160701.iplot(title='before resampling') #before resampling

df_2037160701.rolling(10).mean().iloc[list(np.arange(10,60001,10))].iplot(title='after resampling') #after resampling
import numpy as np



from sklearn.linear_model import LinearRegression

import scipy.stats as spstats





def basic_statistics(t_X, x, s, sensor, postfix=''):

    """Computes basic statistics for the training feature set.

    

    Args:

        t_X (pandas.DataFrame): The feature set being built.

        x (pandas.Series): The signal values.

        s (int): The integer number of the segment.

        postfix (str): The postfix string value.

    Return:

        t_X (pandas.DataFrame): The feature set being built.

    """



    t_X.loc[s, f'{sensor}_sum{postfix}']       = x.sum()

    t_X.loc[s, f'{sensor}_mean{postfix}']      = x.mean()

    t_X.loc[s, f'{sensor}_std{postfix}']       = x.std()

    t_X.loc[s, f'{sensor}_var{postfix}']       = x.var() 

    t_X.loc[s, f'{sensor}_max{postfix}']       = x.max()

    t_X.loc[s, f'{sensor}_min{postfix}']       = x.min()

    t_X.loc[s, f'{sensor}_median{postfix}']    = x.median()

    t_X.loc[s, f'{sensor}_skew{postfix}']      = x.skew()

    t_X.loc[s, f'{sensor}_mad{postfix}']       = x.mad()

    t_X.loc[s, f'{sensor}_kurtosis{postfix}']  = x.kurtosis()



    return t_X







def quantiles(t_X, x, s, sensor, postfix=''):

    """Calculates quantile features for the training feature set.

    Args:

        t_X (pandas.DataFrame): The feature set being built.

        x (pandas.Series): The signal values.

        s (int): The integer number of the segment.

        postfix (str): The postfix string value.

    Return:

        t_X (pandas.DataFrame): The feature set being built.

    """

    t_X.loc[s, f'{sensor}_q999{postfix}']     = np.quantile(x ,0.999)

    t_X.loc[s, f'{sensor}_q99{postfix}']      = np.quantile(x, 0.99)

    t_X.loc[s, f'{sensor}_q95{postfix}']      = np.quantile(x, 0.95)

    t_X.loc[s, f'{sensor}_q87{postfix}']      = np.quantile(x, 0.87)

    t_X.loc[s, f'{sensor}_q13{postfix}']      = np.quantile(x, 0.13)  

    t_X.loc[s, f'{sensor}_q05{postfix}']      = np.quantile(x, 0.05)

    t_X.loc[s, f'{sensor}_q01{postfix}']      = np.quantile(x, 0.01)

    t_X.loc[s, f'{sensor}_q001{postfix}']     = np.quantile(x ,0.001)

    

    x_abs = np.abs(x)

    t_X.loc[s, f'{sensor}_q999_abs{postfix}'] = np.quantile(x_abs, 0.999)

    t_X.loc[s, f'{sensor}_q99_abs{postfix}']  = np.quantile(x_abs, 0.99)

    t_X.loc[s, f'{sensor}_q95_abs{postfix}']  = np.quantile(x_abs, 0.95)

    t_X.loc[s, f'{sensor}_q87_abs{postfix}']  = np.quantile(x_abs, 0.87)

    t_X.loc[s, f'{sensor}_q13_abs{postfix}']  = np.quantile(x_abs, 0.13)

    t_X.loc[s, f'{sensor}_q05_abs{postfix}']  = np.quantile(x_abs, 0.05)

    t_X.loc[s, f'{sensor}_q01_abs{postfix}']  = np.quantile(x_abs, 0.01)

    t_X.loc[s, f'{sensor}_q001_abs{postfix}'] = np.quantile(x_abs, 0.001)

    

    t_X.loc[s, f'{sensor}_iqr']     = np.subtract(*np.percentile(x, [75, 25]))

    t_X.loc[s, f'{sensor}_iqr_abs'] = np.subtract(*np.percentile(x_abs, [75, 25]))



    return t_X





def __linear_regression(arr, abs_v=False):

    """

    """

    idx = np.array(range(len(arr)))

    if abs_v:

        arr = np.abs(arr)

    lr = LinearRegression()

    fit_X = idx.reshape(-1, 1)

    lr.fit(fit_X, arr)

    return lr.coef_[0]





def __classic_sta_lta(x, length_sta, length_lta):

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





def linear_regression(t_X, x, s, sensor, postfix=''):

    t_X.loc[s, f'{sensor}_lr_coef{postfix}'] = __linear_regression(x)

    t_X.loc[s, f'{sensor}_lr_coef_abs{postfix}'] = __linear_regression(x, True)

    return t_X





def classic_sta_lta(t_X, x, sensor, s):

    t_X.loc[s, f'{sensor}_classic_sta_lta1_mean'] = __classic_sta_lta(x, 500, 10000).mean()

    t_X.loc[s, f'{sensor}_classic_sta_lta2_mean'] = __classic_sta_lta(x, 5000, 100000).mean()

    t_X.loc[s, f'{sensor}_classic_sta_lta3_mean'] = __classic_sta_lta(x, 3333, 6666).mean()

    t_X.loc[s, f'{sensor}_classic_sta_lta4_mean'] = __classic_sta_lta(x, 10000, 25000).mean()

    return t_X





def fft(t_X, x, s, sensor, postfix=''):

    """Generates basic statistics over the fft of the signal"""

    z = np.fft.fft(x)

    fft_real = np.real(z)

    fft_imag = np.imag(z)



    t_X.loc[s, f'fft_A0']             = abs(z[0])

    

    t_X.loc[s, f'{sensor}_fft_real_mean{postfix}']      = fft_real.mean()

    t_X.loc[s, f'{sensor}_fft_real_std{postfix}']       = fft_real.std()

    t_X.loc[s, f'{sensor}_fft_real_max{postfix}']       = fft_real.max()

    t_X.loc[s, f'{sensor}_fft_real_min{postfix}']       = fft_real.min()

    t_X.loc[s, f'{sensor}_fft_real_median{postfix}']    = np.median(fft_real)

    t_X.loc[s, f'{sensor}_fft_real_skew{postfix}']      = spstats.skew(fft_real)

    t_X.loc[s, f'{sensor}_fft_real_kurtosis{postfix}']  = spstats.kurtosis(fft_real)

    

    t_X.loc[s, f'{sensor}_fft_imag_mean{postfix}']      = fft_imag.mean()

    t_X.loc[s, f'{sensor}_fft_imag_std{postfix}']       = fft_imag.std()

    t_X.loc[s, f'{sensor}_fft_imag_max{postfix}']       = fft_imag.max()

    t_X.loc[s, f'{sensor}_fft_imag_min{postfix}']       = fft_imag.min()

    t_X.loc[s, f'{sensor}_fft_imag_median{postfix}']    = np.median(fft_imag)

    t_X.loc[s, f'{sensor}_fft_imag_skew{postfix}']      = spstats.skew(fft_imag)

    t_X.loc[s, f'{sensor}_fft_imag_kurtosis{postfix}']  = spstats.kurtosis(fft_imag)

    

    return t_X
train = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv')

train_set = pd.DataFrame()

train_set['segment_id'] = train.segment_id

train_set = train_set.set_index('segment_id')



j = 0

for seg in train.segment_id:

    signals = pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/{seg}.csv')

    signals = signals.rolling(10).mean().iloc[list(np.arange(10,60001,10))]

    for i in range(1, 11):

        sensor_id = f'sensor_{i}'

        train_set = basic_statistics(train_set, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        train_set = quantiles(train_set, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        train_set = linear_regression(train_set, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        train_set = fft(train_set, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        

train_set = pd.merge(train_set.reset_index(), train, on=['segment_id'], how='left').set_index('segment_id')
test = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')

test_set = pd.DataFrame()

test_set['segment_id'] = test.segment_id

test_set = test_set.set_index('segment_id')





for seg in test.segment_id:

    signals = pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/{seg}.csv')

    signals = signals.rolling(10).mean().iloc[list(np.arange(10,60001,10))]

    

    for i in range(1, 11):

        sensor_id = f'sensor_{i}'

        test_set = basic_statistics(test_set, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        test_set = quantiles(test_set, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        test_set = linear_regression(test_set, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        test_set = fft(test_set, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')
import lightgbm as lgbm

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold



y = train_set['time_to_eruption']

feature_df = train_set.drop(['time_to_eruption'], axis = 1)



scaler = StandardScaler()

scaler.fit(feature_df)

scaled_feature_df = pd.DataFrame(scaler.transform(feature_df), columns=feature_df.columns)

scaled_test_df    = pd.DataFrame(scaler.transform(test_set), columns=test_set.columns)



print(scaled_feature_df.shape)

print(scaled_test_df.shape)





n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

scaled_feature_df_columns = scaled_feature_df.columns.values





params = {

    'num_leaves': 85,

    'min_data_in_leaf': 10, 

    'objective':'regression',

    'max_depth': -1,

    'learning_rate': 0.001,

    'max_bins': 2048,

    "boosting": "gbdt",

    "feature_fraction": 0.91,

    "bagging_freq": 1,

    "bagging_fraction": 0.91,

    "bagging_seed": 42,

    "metric": 'mae',

    "lambda_l1": 0.1,

    "verbosity": -1,

    "nthread": -1,

    "random_state": 42

}





oof = np.zeros(len(scaled_feature_df))

predictions = np.zeros(len(scaled_test_df))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_feature_df, y.values)):

    

    strLog = "fold {}".format(fold_)

    print(strLog)

    

    X_tr, X_val = scaled_feature_df.iloc[trn_idx], scaled_feature_df.iloc[val_idx]

    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]



    model = lgbm.LGBMRegressor(**params, n_estimators = 20000, n_jobs = -1)

    model.fit(X_tr, y_tr, 

              eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',

              verbose=1000, early_stopping_rounds=400)

    

    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)



    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = scaled_feature_df_columns

    fold_importance_df["importance"] = model.feature_importances_[:len(scaled_feature_df_columns)]

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += model.predict(scaled_test_df, num_iteration=model.best_iteration_) / folds.n_splits
import matplotlib.pyplot as plt

import seaborn as sns



cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:3014].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,26*3))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()
submission = pd.DataFrame()

submission['segment_id'] = test_set.index

submission['time_to_eruption'] = predictions

submission.to_csv('submission_recent.csv', header=True, index=False)
feature_df.to_pickle('feature_df.pickle')

test_set.to_pickle('test_set.pickle')

scaled_feature_df.to_pickle('scaled_feature_df.pickle')

scaled_test_df.to_pickle('scaled_test_df.pickle')