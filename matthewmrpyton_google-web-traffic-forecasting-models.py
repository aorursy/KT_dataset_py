import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import seaborn as sns

from tsfresh import extract_features

import re

from collections import Counter

from fbprophet import Prophet

import statsmodels.api as sm

from statsmodels.tsa.ar_model import AutoReg

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.seasonal import seasonal_decompose

from matplotlib import pyplot

from datetime import date, timedelta

from scipy import stats

import scipy

from statsmodels.tsa.api import ExponentialSmoothing





np.random.seed(0)



import warnings  

warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/train_2.csv.zip').fillna(0)
def rmsse(train, y, y_hat):

    num = np.sum(np.power((y - y_hat), 2))

    den = (1 / (len(train) - 1)) * np.sum(np.power(np.diff(train), 2))

    return np.sqrt((1 / len(y)) * num  / den)





def smape(y_true, y_pred):

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0

    diff = np.abs(y_true - y_pred) / denominator

    diff[denominator == 0] = 0.0

    return np.nanmean(diff)





def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))





def ts_folds(x, n, h, val_type='exp_wind'):

    assert len(x) - 2 * h > n * h, "horisont or number of folds are too large."

    assert n > 0 and h > 0

    

    l = len(x)

    folds = []

    

    if val_type == 'exp_wind':

        for i in range(n):

            folds.append((x[:l - h*(i + 1)], x[l - h*(i+1):l - h*i]))

    

    return folds





def plot_ts(x_ts, info):

    x = x_ts.values

    

    ts_name = 'X'

    fig, axes = plt.subplots(3, 2, figsize=(18, 20))



    axes[0, 0].set_title(f"Original, id: {info['id']}", fontsize=16)

    axes[0, 0].plot(x, label=ts_name)

    axes[0, 0].set_ylabel('X', fontsize=16)

    axes[0, 0].set_xlabel('t', fontsize=16)

    axes[0, 0].legend()



    plot_acf(x, lags=90, ax=axes[0, 1])



    sns.distplot(x, bins=100, kde=True, ax=axes[1, 0])

    axes[1, 0].set_xlabel('X', fontsize=16)

    axes[1, 0].set_title('Histogram', fontsize=16)



    x_ts.diff().plot(ax=axes[1, 1], title='1 step difference')



    axes[2, 0].set_title('Box plot (grouped by months)', fontsize=16)

    sns.boxplot(x_ts.index.month, x_ts, ax=axes[2, 0])



    axes[2, 1].plot(x_ts, label=ts_name)

    x_ts.rolling(window=7).mean().plot(ax=axes[2, 1], label='W=7')

    x_ts.rolling(window=30).mean().plot(ax=axes[2, 1], label='W=30')

    x_ts.rolling(window=60).mean().plot(ax=axes[2, 1], label='W=60')

    axes[2, 1].legend()

    axes[2, 1].set_title('Moving average', fontsize=16)

    axes[2, 1].set_ylabel('X', fontsize=16)

    axes[2, 1].set_xlabel('t', fontsize=16)



    plt.show()

    

    res = seasonal_decompose(x_ts, model='additive')

    

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 15))

    fig.suptitle('additive decomposition')

    res.observed.plot(ax=ax1)

    ax1.set_ylabel('original')

    res.trend.plot(ax=ax2)

    ax2.set_ylabel('trend')

    res.seasonal.plot(ax=ax3)

    ax3.set_ylabel('seasonal')

    res.resid.plot(ax=ax4)

    ax4.set_ylabel('residuals')



    pass

    # return fig, axes





def plot_prediction(train, val, y_hat, plot_len=None, title=None):

    if plot_len == None:

        plot_len = len(train) + len(val)

    

    l = len(train)

    train_range = np.arange(l)

    val_range = np.arange(l, l + len(val))

    all_range = np.arange(l + len(val))



    fig = plt.figure(figsize=(14, 8))

    plt.plot(train_range, train, '.-b', label='train')

    plt.plot(val_range, val, '.-g', label='val')

    plt.plot(val_range, y_hat, '.-r', label='forecast')

    plt.xlim([all_range[-plot_len:][0], all_range[-plot_len:][-1]])

    plt.legend()

    plt.vlines(l, -1e6, 1e6, 'k', 'dashed')

    plt.ylim([np.min([np.min(train[-plot_len:]), np.min(val)])-1, np.max([np.max(train[-plot_len:]), np.max(val)])+10])

    plt.title(title)

    plt.show()





class TransformTS():

    def __init__(self, tr_type, handle_missing=None):

        self.tr_type = tr_type

        self.handle_missing = handle_missing

    

    def transform(self, x):

        if self.tr_type == 'ln_p1':

            y = np.log(x + 1)

        elif self.tr_type == 'ln':

            y = np.log(x)

        elif self.tr_type == 'sqrt':

            y = np.sqrt(x)

        elif self.tr_type == 'boxcox':

            y, lb = stats.boxcox(x)

            self.lb = lb

        else:

            raise ValueError

            

        return y

    

    def reverse(self, x):

        if self.tr_type == 'ln_p1':

            y = np.exp(x) - 1

        elif self.tr_type == 'ln':

            y = np.exp(x)

        elif self.tr_type == 'sqrt':

            y = np.power(x, 2)

        elif self.tr_type == 'boxcox':

            y = scipy.special.inv_boxcox(x, self.lb)

        else:

            raise ValueError

        

        return y

# Models.

HORIZONT = 64





def prophet_fc(x, h=HORIZONT):

    # Prophet model.

    # pr_df = pd.DataFrame({'y': x['y'], 'ds': x['ds']})

    m = Prophet()

    m.fit(x)

    future = m.make_future_dataframe(periods=h)

    forecast = m.predict(future)

    y_hat = forecast['yhat'][-h:]

    return y_hat





def naive(x, h=HORIZONT):

    return np.ones(h) * np.mean(x[-HORIZONT:])





def naive2(x, h=HORIZONT):

    return np.ones(h) * x[-1]





def arima_mdl(x, order, h=HORIZONT):

    p, d, q = order

    l = len(x)



    mod = sm.tsa.statespace.SARIMAX(x, order=(p, d, q), trend='c').fit(disp=False)

    y_hat = mod.forecast(steps=HORIZONT)



    return y_hat





def agg_weekly(x, window=64, h=HORIZONT, avg_type='mean'):

    xs = x[-window:]

    

    # Median grouped by weekday.

    medians = [xs[xs.index.weekday == i].median() for i in range(7)]

    means = [xs[xs.index.weekday == i].mean() for i in range(7)]

    

    d = x.index[-1].weekday()

    y_hat = np.zeros(HORIZONT)



    for n in range(HORIZONT):

        if d == 6:

            d = 0

        else:

            d = d + 1

        

        if avg_type == 'median':

            y_hat[n] = medians[d]

        elif avg_type == 'mean':

            y_hat[n] = means[d]

        else:

            raise ValueError



    return y_hat





def exp_smoothing(x_train, config, h=HORIZONT):

    t,d,s,p,b,r = config



    model = ExponentialSmoothing(x_train, trend=t, damped=d, seasonal=s, seasonal_periods=p)

    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)

    yhat = model_fit.forecast(h)

    

    return yhat

# An example of the time series and Naive forecast.



i = 100

x_df = train.iloc[i]

x_ts = pd.Series(x_df.values[1:].astype(float), index=x_df.index[1:])

x_ts.index = pd.to_datetime(x_ts.index)

x = x_ts.values



folds = ts_folds(x, n=1, h=HORIZONT)



x_train = folds[0][0]

y_val = folds[0][1]



y_hat = naive(x_train)

plot_prediction(x_train, y_val, y_hat, plot_len=400, title=f"naive forecast, SMAPE={smape(y_val, y_hat):.3f}")
# Prophet model.



x_df = pd.DataFrame({'ds': x_ts.index, 'y': x_ts.values})

m = Prophet()



m.fit(x_df)

future = m.make_future_dataframe(periods=HORIZONT)

forecast = m.predict(future)

y_hat = forecast['yhat'][-HORIZONT:]

y_hat[y_hat < 0] = 0



plot_prediction(x_train, y_val, y_hat, plot_len=400, title=f"prophet, SMAPE={smape(y_val, y_hat):.3f}")

fig1 = m.plot(forecast)
# Config for ETS Models.

t_params = ['add', 'mul', None]

d_params = [True, False]

s_params = ['add', 'mul', None]

p_params = [5, 7, 12, 14, 30]

b_params = [True, False]

r_params = [True, False]



cfg_list = []

for t in t_params:

    for d in d_params:

        for s in s_params:

            for p in p_params:

                for b in b_params:

                    for r in r_params:

                        if d is True and t is None:

                            continue

                        cfg = [t, d, s, p, b, r]

                        cfg_list.append(cfg)



# print(f'{len(cfg_list)} combinations')
# List of models.



def eval_models(train, y_val, m_list):

    

    models = []



    # Naive1.

    if 'naive1' in m_list:

        y_hat = naive(train.values, h=HORIZONT)

        y_hat[y_hat < 0] = 0

        models.append({

            'name': 'naive1',

            'y_pred': y_hat,

            'smape': smape(y_val, y_hat),

            'rmse': np.sqrt(mean_squared_error(y_hat, y_val)),

            'model': None}

        )



    # Naive2.

    if 'naive2' in m_list:

        y_hat = naive2(train.values, h=HORIZONT)

        y_hat[y_hat < 0] = 0

        models.append({

            'name': 'naive2',

            'y_pred': y_hat,

            'smape': smape(y_val, y_hat),

            'rmse': np.sqrt(mean_squared_error(y_hat, y_val)),

            'model': None}

        )

    

    # Prophet.

    if 'prophet' in m_list:

        x_df = pd.DataFrame({'ds': train.index, 'y': x_train.values})

        y_hat = prophet_fc(x_df, h=HORIZONT)

        y_hat[y_hat < 0] = 0

        models.append({

            'name': 'prophet',

            'y_pred': y_hat,

            'smape': smape(y_val, y_hat),

            'rmse': np.sqrt(mean_squared_error(y_hat, y_val)),

            'model': None}

        )

    

    # ARIMA models.

    for p in range(30):

        for d in [0, 1]:

            for q in range(4):

                m_name = f'arima_{p}_{d}_{q}'

        

                if m_name in m_list:

                    try:

                        y_hat = arima_mdl(train, (p, d, q), h=HORIZONT)

                        conv = True

                    except:

                        y_hat = naive(train.values, h=HORIZONT)

                        conv = False

                    y_hat[y_hat < 0] = 0

                    models.append({

                        'name': m_name,

                        'y_pred': y_hat,

                        'smape': smape(y_val, y_hat),

                        'rmse': np.sqrt(mean_squared_error(y_hat, y_val)),

                        'model': None,

                        'conv':  conv,

                    })

    

    # Average by weekdays.

    winds = [7, 32, 64, 96]

    for w in winds:

        if f'weekly_mean_w-{w}' in m_list:

            y_hat = agg_weekly(train, window=w, h=HORIZONT, avg_type='mean')

            y_hat[y_hat < 0] = 0

            models.append({

                'name': f'weekly_mean_w-{w}',

                'y_pred': y_hat,

                'smape': smape(y_val, y_hat),

                'rmse': np.sqrt(mean_squared_error(y_hat, y_val)),

                'model': None}

            )

    

    winds = np.arange(1, 13) * 7

    for w in winds:

        if f'weekly_median_w-{w}' in m_list:

            y_hat = agg_weekly(train, window=w, h=HORIZONT, avg_type='median')

            y_hat[y_hat < 0] = 0

            models.append({

                'name': f'weekly_median_w-{w}',

                'y_pred': y_hat,

                'smape': smape(y_val, y_hat),

                'rmse': np.sqrt(mean_squared_error(y_hat, y_val)),

                'model': None}

            )



    

    # ETS models.

    for cfg in cfg_list:

        if f"ets_{cfg}" in m_list:

            

            # remove zeros for box-cox and mult trend.

            train = train + 1

            

            y_hat = exp_smoothing(train, cfg, h=HORIZONT)

            

            if y_hat.isnull().values.any() or np.isinf(y_hat.values).any():

                y_hat = naive(train.values, h=HORIZONT)

                conv = False

            else:

                conv = True

            

            # reverse transformation.

            y_hat = y_hat - 1

            y_hat[y_hat < 0] = 0



            models.append({

                'name': f'ets_{cfg}',

                'y_pred': y_hat,

                'smape': smape(y_val, y_hat),

                'rmse': np.sqrt(mean_squared_error(y_hat, y_val)),

                'model': None,

                'conv':  conv,}

            )

        

    return models

# Cross validation, N-fold expanding window.



N_FOLDS = 1

MAX_EVAL = 2200



power_trans = True



metr_arr = []

pred_arr = []



# Models to evaluate.

m_list = [

    'naive1',

    # 'naive2',

    'prophet',

    'weekly_median_w-28',

    'weekly_median_w-35',

    'weekly_median_w-42',

    'weekly_median_w-49',

    'weekly_median_w-56',

    'weekly_median_w-63',

    "ets_['add', True, 'add', 5, True, False]",

    "ets_['add', True, 'add', 7, True, False]",

    "ets_['add', True, 'add', 12, True, False]",

    "ets_['add', True, 'add', 14, True, False]",

    'arima_6_1_1',

]



# ETS models.

# for cfg in cfg_list:

#     m_list.append(f"ets_{cfg}")

    

np.random.seed(0)

idx = np.arange(train.shape[0])

ind_list = np.random.choice(idx, size=MAX_EVAL, replace=False)





for k, i in enumerate(ind_list):

    if k % 1000 == 0:

        print('step:', k)



    x_df = train.iloc[i]

    x_ts = pd.Series(x_df.values[1:].astype(float), index=x_df.index[1:])

    x_ts.index = pd.to_datetime(x_ts.index)



    folds = ts_folds(x_ts, n=N_FOLDS, h=HORIZONT)

    

    for n in range(N_FOLDS):

        x_train = folds[n][0]

        y_val = folds[n][1].values



        # Check if last 64 values are zero. Predict with zeros.

        if np.sum(x_train[-64:].values) == 0:

            print('zeros')

            y_hat = np.zeros(HORIZONT)

            models = []

            for name in m_list: 

                models.append({

                'name': name,

                'smape': smape(y_val, y_hat),

                'y_pred': y_hat,

                'rmse': np.sqrt(mean_squared_error(y_hat, y_val))

                })

        else:

            models = eval_models(x_train, y_val, m_list)

        

        for m in models:

            x = {

                'id': i,

                'fold_num': n,

                'name': m['name'],

                'smape': m['smape'],

                'rmse': m['rmse']

            }

            

            metr_arr.append(x)

            

            pred_arr.append({

                'id': i,

                'fold_num': n,

                'smape': m['smape'],

                'rmse': m['rmse'],

                'name': m['name'],

                'y_hat': m['y_pred'],

                'x_train': x_train,

                'y_val': y_val

            })



    if k == MAX_EVAL:

        break



metr_df = pd.DataFrame(metr_arr)

pred_df = pd.DataFrame(pred_arr)
# Plotting metrics.

# Adjusted by "naive1" for each id and each fold.



for i in metr_df['id'].unique():

    for n in metr_df['fold_num'].unique():

        d = metr_df[(metr_df['id'] == i) & (metr_df['fold_num'] == n)]

        metr_df.loc[(metr_df['id'] == i) & (metr_df['fold_num'] == n), 'smape_adj'] = d['smape'] / d[d['name'] == 'naive1']['smape'].values[0]

        metr_df.loc[(metr_df['id'] == i) & (metr_df['fold_num'] == n), 'rmse_adj'] = d['rmse'] / d[d['name'] == 'naive1']['rmse'].values[0]

# Plot ungrouped. SMAPE adj.



def plot_metr(metr, df):

        

    arr = m_list[:]

    # arr.remove('naive1')



    for n, mod_name in enumerate(arr):

        n = 0

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        x = df[df['name'] == mod_name][metr].values



        sns.distplot(x, bins=40, kde=True, ax=axes[0])

        axes[0].set_title(f'Model: {mod_name}', fontsize=14)

        axes[0].set_xlabel(metr, fontsize=14)



        axes[1].plot(x, '.')

        axes[1].set_ylabel(metr, fontsize=14)

        # axes[1].plot([1]*len(x), '--k')

        axes[1].set_title(f'{metr} MEAN = {np.mean(x):.5f}', fontsize=14)

        plt.show()

probl_ids = metr_df[metr_df['smape_adj'].isnull()]['id'].unique()

# cleaning

df = metr_df.replace([np.inf, -np.inf], np.nan)

df = df.dropna()
# metr = 'smape_adj'

metr = 'smape'

plot_metr(metr, df)
# Table of metrics.

print(f'Number of series: {MAX_EVAL}')

agg_df = df.groupby('name').mean().drop(['id', 'fold_num'], axis=1)

display(agg_df)
# Plotting outliers for some model.



# m_name = 'weekly_median_w-63'

m_name = agg_df['smape'].idxmin()

print('Models name:', m_name)



# Inspect all that higher than 60.

smape_cut = 60

fold_num = 0



ids = df[(df['smape'] > smape_cut) & (df['name'] == m_name) & (df['fold_num'] == fold_num)]['id'].unique()

i = ids[0]



# Plot several from the list.

for i in ids[:5]:

    x = pred_df[(pred_df['id'] == i) & (pred_df['fold_num'] == fold_num) & (pred_df['name'] == m_name)]

    smape_val = x['smape'].values[0]

    y_hat = x['y_hat'].values[0]

    y_val = x['y_val'].values[0]

    x_train = x['x_train'].values[0]

    

    plot_prediction(x_train, y_val, y_hat, plot_len=400, title=f"Id: {i}, SMAPE: {smape_val}")

# Plot several good fits.

m_name = agg_df['smape'].idxmin()

print('Models name:', m_name)



# Inspect all that higher than 60.

fold_num = 0



ids = df[(df['smape'] > 30) & (df['smape'] < 37) & (df['name'] == m_name) & (df['fold_num'] == fold_num)]['id'].unique()

i = ids[0]



# Plot several from the list.

for i in ids[:5]:

    x = pred_df[(pred_df['id'] == i) & (pred_df['fold_num'] == fold_num) & (pred_df['name'] == m_name)]

    smape_val = x['smape'].values[0]

    y_hat = x['y_hat'].values[0]

    y_val = x['y_val'].values[0]

    x_train = x['x_train'].values[0]

    

    plot_prediction(x_train, y_val, y_hat, plot_len=400, title=f"Id: {i}, SMAPE: {smape_val}")

# Building the submission on isolated models.



subm = True

# subm = False





if subm:

    print('Building the submission...')

    

    keys = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/key_2.csv.zip')

    keys['date'] = keys['Page'].apply(lambda x: x.split("_")[-1])

    keys['name'] = keys['Page'].apply(lambda x: '_'.join(x.split("_")[:-1]))

    keys['horisont_day'] = None

    sdate = date(2017, 9, 13)

    edate = date(2017, 11, 13)

    delta = edate - sdate

    dates_list = [str(sdate + timedelta(days=i)) for i in range(delta.days + 1)] 

    for n, d in enumerate(dates_list):

        keys.loc[keys['date'] == d, 'horisont_day'] = n

    

    # One model on which to build the submission.

    m_list = ['weekly_median_w-49']

    # m_list = ["ets_['add', True, 'add', 7, True, False]"]

    

    res = []



    print('Predicting...')

    for k, i in enumerate(np.arange(train.shape[0])):

        if k % 10000 == 0:

            print('Step:', k)



        x_df = train.iloc[i]

        page = x_df['Page']

        

        x_ts = pd.Series(x_df.values[1:].astype(float), index=x_df.index[1:])

        x_ts.index = pd.to_datetime(x_ts.index)



        # Check if last 50 values are zero. If so, predict with zeros.

        if np.sum(x_ts[-50:].values) == 0:

            y_hat = np.zeros(HORIZONT)

        else:

            models = eval_models(x_ts, np.ones(HORIZONT), m_list)

            y_hat = models[0]['y_pred']

        

        # 62 last days out of 64.

        y_hat = y_hat[2:]

        for m, y in enumerate(y_hat):

            res.append({'name': page, 'horisont_day': m, 'Visits': y})



    res_df = pd.DataFrame(res)



    subm = res_df.merge(keys, how='left', on=['name', 'horisont_day'])

    subm['Visits'] = subm['Visits'].map(np.round).astype(int)

    subm_df = subm[['Id', 'Visits']].sort_values(by='Id')

    # subm_df.head()

    subm_df.to_csv('subm.csv', encoding='utf-8', index=False)
