from statsmodels.tsa.ar_model import AutoReg, ar_select_order

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.api import acf, pacf, graphics

from typing import List, Tuple, Union, NoReturn

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

import cufflinks as cf

import plotly

from statsmodels.robust import mad

import matplotlib.pyplot as plt

from scipy.signal import butter

from scipy import signal

import lightgbm as lgb

import seaborn as sns

from sklearn import *

import pandas as pd 

import numpy as np

import warnings

import scipy

import pywt

import os



cf.go_offline()

py.init_notebook_mode()

cf.getThemes()

cf.set_config_file(theme='ggplot')

warnings.simplefilter('ignore')

pd.plotting.register_matplotlib_converters()

sns.mpl.rc('figure',figsize=(16, 6))

plt.style.use('ggplot')

sns.set_style('darkgrid')

base = os.path.abspath('/kaggle/input/liverpool-ion-switching/')

train = pd.read_csv(os.path.join(base + '/train.csv'))

test  = pd.read_csv(os.path.join(base + '/test.csv'))
train.shape[0], test.shape[0]
train.head()
train.describe()
test.head()
def add_bathing_to_data(df : pd.DataFrame) -> pd.DataFrame :

    batches = df.shape[0] // 500000

    df['batch'] = 0

    for i in range(batches):

        idx = np.arange(i*500000, (i+1)*500000)

        df.loc[idx, 'batch'] = i + 1

    return df



def p5( x : pd.Series) -> pd.Series : return x.quantile(0.05)

def p95(x : pd.Series) -> pd.Series : return x.quantile(0.95)

train = add_bathing_to_data(train)

train.groupby('batch')[['signal','open_channels']].agg(['min', 'max', 'median', p5, p95])

train.groupby('open_channels')[['signal','batch']].agg(['min', 'max', 'median', p5, p95])

train.groupby(['batch','open_channels'])[['signal']].agg(['min', 'max', 'median', p5, p95])

partial = train.iloc[::250, :]

partial.signal = np.round(partial.signal.values, 2)

partial['shifted_signal'] = (partial.signal.values + 10) ** 2

fig = px.scatter(partial, x='signal', y='open_channels', color='open_channels',size='shifted_signal',  title='Signal vs Channels')

fig.show()

fig = px.box(partial, x='open_channels', y='signal', color='open_channels', title='Signal vs Channels')

fig.update_traces(quartilemethod='exclusive')

fig.show()
fig = px.box(partial, x='open_channels', y='signal', color='batch', title='Signal vs Channels for Batches')

fig.update_traces(quartilemethod='exclusive')

fig.show()
fig = px.density_heatmap(train.iloc[::50, :], x='signal', y='open_channels')

fig.show()

fig = make_subplots(rows=5, cols=2,  subplot_titles=[f'Batch no {i+1}' for i in range(10)])

i = 1

for row in range(1, 6):

    for col in range(1, 3):

        data = train[train.batch==i]['open_channels'].value_counts(sort=False).values

        fig.add_trace(go.Bar(x=list(range(11)), y=data), row=row, col=col)       

        i += 1

fig.update_layout(width=800, height=1500, title_text="Target for each batch", showlegend=False)

train.open_channels.value_counts(sort=False).iplot(kind='bar')
def plot_by_batch_summaries(df : pd.DataFrame) -> NoReturn :

    by_batch = df.groupby(['batch']).agg(['min', 'max', 'median', p5, p95]).reset_index(drop=True).iloc[:,5:]

    by_batch.columns = ['MIN-SIG','MAX-SIG', 'MED-SIG', '5P-SIG', '95P-SIG', 'MIN-CHANNEL','MAX-CHANNEL', 'MED-CHANNEL', '5P-CHANNEL', '95P-CHANNEL']

    by_batch.iloc[:,:5].iplot(kind='bar',xTitle='Batch', yTitle='Signal')

    by_batch.iloc[:,5:].iplot(kind='bar', xTitle='Batch', yTitle='Channel')



plot_by_batch_summaries(train)
def plot_by_channel_summaries(df : pd.DataFrame) -> NoReturn :

    by_channel = train.groupby(['open_channels']).agg(['min', 'max', 'median', p5, p95]).reset_index(drop=True).iloc[:,5:]

    by_channel.columns = ['MIN-SIG','MAX-SIG', 'MED-SIG', '5P-SIG', '95P-SIG', 'MIN-BATCH','MAX-BATCH', 'MED-BATCH', '5P-BATCH', '95P-BATCH']

    by_channel.iloc[:,5:].iplot(kind='bar' ,xTitle='Channel', yTitle='Batch')

    by_channel.iloc[:,:5].iplot(kind='bar' )



plot_by_channel_summaries(train)
def plot_by_channel_summaries(df : pd.DataFrame, resample : int) -> NoReturn :

    train_resampled = df.iloc[::resample, :]

    train_resampled[['signal','open_channels']].plot(subplots=True)

    plt.show()

    

plot_by_channel_summaries(train, 10000)

def plot_smoothed_batch(i : int, window : int) -> NoReturn:

    batch_resampled = train[train.batch==i].iloc[::1000, :]

    ts = batch_resampled['signal']

    plt.plot(ts, 'r-', color='royalblue')

    plt.ylabel('Signal')

    smooth_data = pd.Series(ts).rolling(window=window).mean().plot(style='k')

    plt.show()
def maddest(d, axis=None):

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)



def high_pass_filter(x, low_cutoff=1000, sample_rate=10000):



    nyquist = 0.5 * sample_rate

    norm_low_cutoff = low_cutoff / nyquist

    print(norm_low_cutoff)

    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')

    filtered_sig = signal.sosfilt(sos, x)



    return filtered_sig



def denoise_signal( x, wavelet='db4', level=1):

    

    coeff = pywt.wavedec( x, wavelet, mode="per" )

    sigma = (1/0.6745) * maddest( coeff[-level] )

    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )

    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )

    return pywt.waverec( coeff, wavelet, mode='per' )
def plot_acf_pacf(i : int, lag : int, resample : int) -> NoReturn:

    batch_resampled = train[train.batch==i].iloc[::resample, :]

    plot_acf( batch_resampled['signal'], lags=lag)

    plot_pacf(batch_resampled['signal'], lags=lag)

    plt.show()
def add_rooling_data(df : pd.DataFrame) -> pd.DataFrame:

    window_sizes = [10, 50, 100, 1000]

    for window in window_sizes:

        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()

        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()

    return df
train = add_rooling_data(train)
def plot_rolling_window(i : int, resample : int) -> NoReturn:

    window_sizes = [10, 50, 100, 1000]

    fig, ax = plt.subplots(len(window_sizes),1,figsize=(20, 6 * len(window_sizes)))

    n = 0

    for col in train.columns.values:

        if 'rolling_' in col:

            if 'mean' in col:

                mean_df = train[train.batch==i].iloc[::resample,:][col]

                ax[n].plot(mean_df, label=col, color='navy')

            if 'std' in col:

                std = train[train.batch==i].iloc[::resample,:][col].values

                ax[n].fill_between(mean_df.index.values,

                               mean_df.values-std, mean_df.values+std,

                               facecolor='lightskyblue',

                               alpha = 0.5, label=col)

                ax[n].legend()

                n+=1
def plot_batch(i : int, resample : int) -> NoReturn:

    batch_resampled = train[train.batch==i].iloc[::resample, :]

    batch_resampled[['signal','open_channels']].plot(subplots=True)

    plt.show()

    ax = sns.distplot(batch_resampled[['signal']], rug=True)

    ax.set_title(f'  Signal Distribution Batch=={i}', fontsize=13)

    mod = AutoReg(batch_resampled['signal'], 3)

    res = mod.fit(cov_type="HC0")

    sel = ar_select_order(batch_resampled['signal'], 3, glob=True)

    sel.ar_lags

    res = sel.model.fit()

    fig = plt.figure(figsize=(16,9))

    fig = res.plot_diagnostics(fig=fig, lags=25)

    plot_rolling_window(i, resample)

    return None
def plot_signal_distribution_by_target(i : int) -> NoReturn :

    data_by_target = train[train.open_channels==0]

    ax = sns.distplot(data_by_target[['signal']], rug=True)

    ax.set_title(f'Signal Distribution for Target=={i}', fontsize=13)

    plt.show()

    return None
def plot_denoided_batch(i : int, resample : int) -> NoReturn : 

    batch_resampled = train[train.batch==i].iloc[::5000, :]  

    batch_resampled['x_dn_1'] = denoise_signal(batch_resampled['signal'], wavelet='db4', level=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=batch_resampled.time, mode='lines+markers', y=batch_resampled.signal, marker=dict(color="lightskyblue"), name="Original signal"))

    fig.add_trace(go.Scatter(x=batch_resampled.time, y=batch_resampled.x_dn_1, mode='lines', marker=dict(color="navy"), name="Denoised signal"))

    fig.show()
plot_batch(1, resample=1000)
plot_smoothed_batch(1, window=25)
plot_denoided_batch(1, resample=5000)
plot_acf_pacf(1, lag=25, resample=5000)
plot_signal_distribution_by_target(0)
base = os.path.abspath('/kaggle/input/liverpool-ion-switching/')

train = pd.read_csv(os.path.join(base + '/train.csv'))

test  = pd.read_csv(os.path.join(base + '/test.csv'))



def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    df['batch_slices'] = df['batch_index']  // 2500

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal'].mean()

        d['median'+c] = df.groupby([c])['signal'].median()

        d['max'+c] = df.groupby([c])['signal'].max()

        d['min'+c] = df.groupby([c])['signal'].min()

        d['std'+c] = df.groupby([c])['signal'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        d['range'+c] = d['max'+c] - d['min'+c]

        d['maxtomin'+c] = d['max'+c] / d['min'+c]

        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    

    #add shifts

    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])

    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    return df



train = features(train)

test = features(test)



col = [c for c in train.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]

x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['open_channels'], test_size=0.3, random_state=7)



def MacroF1Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = metrics.f1_score(labels, preds, average = 'macro')

    return ('MacroF1Metric', score, True)





params = {'learning_rate': 0.1, 'max_depth': -1, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1, 'sample_fraction':0.33} 

model = lgb.train(params, lgb.Dataset(x1, y1), 2000,  lgb.Dataset(x2, y2), verbose_eval=0, early_stopping_rounds=100, feval=MacroF1Metric)

train_preds = model.predict(train[col], num_iteration=model.best_iteration)

preds = model.predict(test[col], num_iteration=model.best_iteration)
from functools import partial

import scipy as sp

class OptimizedRounder(object):

    """

    An optimizer for rounding thresholds

    to maximize F1 (Macro) score

    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved

    """

    def __init__(self):

        self.coef_ = 0



    def _f1_loss(self, coef, X, y):

        """

        Get loss according to

        using current coefficients

        

        :param coef: A list of coefficients that will be used for rounding

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



        return -metrics.f1_score(y, X_p, average = 'macro')



    def fit(self, X, y):

        """

        Optimize rounding thresholds

        

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        loss_partial = partial(self._f1_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        """

        Make predictions with specified thresholds

        

        :param X: The raw predictions

        :param coef: A list of coefficients that will be used for rounding

        """

        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])





    def coefficients(self):

        """

        Return the optimized coefficients

        """

        return self.coef_['x']
optR = OptimizedRounder()

optR.fit(train_preds.reshape(-1,), train['open_channels'])

coefficients = optR.coefficients()

print(coefficients)
def optimize_prediction(prediction):

    prediction[prediction <= coefficients[0]] = 0

    prediction[np.where(np.logical_and(prediction > coefficients[0], prediction <= coefficients[1]))] = 1

    prediction[np.where(np.logical_and(prediction > coefficients[1], prediction <= coefficients[2]))] = 2

    prediction[np.where(np.logical_and(prediction > coefficients[2], prediction <= coefficients[3]))] = 3

    prediction[np.where(np.logical_and(prediction > coefficients[3], prediction <= coefficients[4]))] = 4

    prediction[np.where(np.logical_and(prediction > coefficients[4], prediction <= coefficients[5]))] = 5

    prediction[np.where(np.logical_and(prediction > coefficients[5], prediction <= coefficients[6]))] = 6

    prediction[np.where(np.logical_and(prediction > coefficients[6], prediction <= coefficients[7]))] = 7

    prediction[np.where(np.logical_and(prediction > coefficients[7], prediction <= coefficients[8]))] = 8

    prediction[np.where(np.logical_and(prediction > coefficients[8], prediction <= coefficients[9]))] = 9

    prediction[prediction > coefficients[9]] = 10

    

    return prediction
test['open_channels'] = optimize_prediction(preds).astype(int)

test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')