import time

import numpy as np

import pandas as pd

from boruta import BorutaPy

# Plotly for Visualizations

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

# Sklearn

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold, train_test_split

from sklearn.linear_model import LinearRegression, LassoCV

from sklearn.ensemble import RandomForestRegressor

# scipy (feature engineering)

from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats

import lightgbm as lgb

import warnings

# Configurations

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=UserWarning)

warnings.simplefilter(action='ignore', category=RuntimeWarning)

init_notebook_mode(connected=True)

RANDOM_SEED = 19

np.random.seed(RANDOM_SEED)



def lineplot(x_axis, y_axis, title=''):

    trace = go.Scatter(x=x_axis, y=y_axis, mode='lines+markers')



    layout = go.Layout(

        title=title, 

        showlegend=False,

        xaxis=dict(

            title='Number of features',

            titlefont=dict(size=14, color='#7f7f7f')

        ),

        yaxis=dict(

            title='Validation score',

            titlefont=dict(size=14, color='#7f7f7f')

        )

    )

    fig = go.Figure(data=[trace], layout=layout)

    iplot(fig)
data_type = {'acoustic_data': np.int16, 'time_to_failure': np.float64}

train = pd.read_csv('../input/train.csv', dtype=data_type)

train.head(3)
rows = 150_000

segments = int(np.floor(train.shape[0] / rows))



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



X_tr = pd.DataFrame(index=range(segments), dtype=np.float64)



y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])



total_mean = train['acoustic_data'].mean()

total_std = train['acoustic_data'].std()

total_max = train['acoustic_data'].max()

total_min = train['acoustic_data'].min()

total_sum = train['acoustic_data'].sum()

total_abs_sum = np.abs(train['acoustic_data']).sum()



def calc_change_rate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)



for segment in range(segments):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = pd.Series(seg['acoustic_data'].values)

    y = seg['time_to_failure'].values[-1]

    

    y_tr.loc[segment, 'time_to_failure'] = y

    X_tr.loc[segment, 'mean'] = x.mean()

    X_tr.loc[segment, 'std'] = x.std()

    X_tr.loc[segment, 'max'] = x.max()

    X_tr.loc[segment, 'min'] = x.min()

    

    X_tr.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))

    X_tr.loc[segment, 'mean_change_rate'] = calc_change_rate(x)

    X_tr.loc[segment, 'abs_max'] = np.abs(x).max()    

    X_tr.loc[segment, 'std_first_50000'] = x[:50000].std()

    X_tr.loc[segment, 'std_last_50000'] = x[-50000:].std()

    X_tr.loc[segment, 'std_first_10000'] = x[:10000].std()

    X_tr.loc[segment, 'std_last_10000'] = x[-10000:].std()

    

    X_tr.loc[segment, 'avg_first_50000'] = x[:50000].mean()

    X_tr.loc[segment, 'avg_last_50000'] = x[-50000:].mean()

    X_tr.loc[segment, 'avg_first_10000'] = x[:10000].mean()

    X_tr.loc[segment, 'avg_last_10000'] = x[-10000:].mean()

    

    X_tr.loc[segment, 'min_first_50000'] = x[:50000].min()

    X_tr.loc[segment, 'min_last_50000'] = x[-50000:].min()

    X_tr.loc[segment, 'min_first_10000'] = x[:10000].min()

    X_tr.loc[segment, 'min_last_10000'] = x[-10000:].min()

    

    X_tr.loc[segment, 'max_first_50000'] = x[:50000].max()

    X_tr.loc[segment, 'max_last_50000'] = x[-50000:].max()

    X_tr.loc[segment, 'max_first_10000'] = x[:10000].max()

    X_tr.loc[segment, 'max_last_10000'] = x[-10000:].max()

    

    X_tr.loc[segment, 'max_to_min'] = x.max() / np.abs(x.min())

    X_tr.loc[segment, 'max_to_min_diff'] = x.max() - np.abs(x.min())

    X_tr.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])

    X_tr.loc[segment, 'sum'] = x.sum()

    

    X_tr.loc[segment, 'mean_change_rate_first_50000'] = calc_change_rate(x[:50000])

    X_tr.loc[segment, 'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])

    X_tr.loc[segment, 'mean_change_rate_first_10000'] = calc_change_rate(x[:10000])

    X_tr.loc[segment, 'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

    

    X_tr.loc[segment, 'q95'] = np.quantile(x, 0.95)

    X_tr.loc[segment, 'q99'] = np.quantile(x, 0.99)

    X_tr.loc[segment, 'q05'] = np.quantile(x, 0.05)

    X_tr.loc[segment, 'q01'] = np.quantile(x, 0.01)

    

    X_tr.loc[segment, 'abs_q95'] = np.quantile(np.abs(x), 0.95)

    X_tr.loc[segment, 'abs_q99'] = np.quantile(np.abs(x), 0.99)

    X_tr.loc[segment, 'abs_q05'] = np.quantile(np.abs(x), 0.05)

    X_tr.loc[segment, 'abs_q01'] = np.quantile(np.abs(x), 0.01)

    

    X_tr.loc[segment, 'trend'] = add_trend_feature(x)

    X_tr.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)

    X_tr.loc[segment, 'abs_mean'] = np.abs(x).mean()

    X_tr.loc[segment, 'abs_std'] = np.abs(x).std()

    

    X_tr.loc[segment, 'mad'] = x.mad()

    X_tr.loc[segment, 'kurt'] = x.kurtosis()

    X_tr.loc[segment, 'skew'] = x.skew()

    X_tr.loc[segment, 'med'] = x.median()

    

    X_tr.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()

    X_tr.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()

    X_tr.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()

    X_tr.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()

    X_tr.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()

    X_tr.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()

    X_tr.loc[segment, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()

    X_tr.loc[segment, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()

    X_tr.loc[segment, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()

    X_tr.loc[segment, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

    X_tr.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    X_tr.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)

    X_tr.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)

    X_tr.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)

    no_of_std = 3

    X_tr.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()

    X_tr.loc[segment,'MA_700MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()

    X_tr.loc[segment,'MA_700MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()

    X_tr.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()

    X_tr.loc[segment,'MA_400MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()

    X_tr.loc[segment,'MA_400MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()

    X_tr.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()

    X_tr.drop('Moving_average_700_mean', axis=1, inplace=True)

    

    X_tr.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    X_tr.loc[segment, 'q999'] = np.quantile(x,0.999)

    X_tr.loc[segment, 'q001'] = np.quantile(x,0.001)

    X_tr.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)



    for windows in [10, 100, 1000]:

        x_roll_std = x.rolling(windows).std().dropna().values

        x_roll_mean = x.rolling(windows).mean().dropna().values

        

        X_tr.loc[segment, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X_tr.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()

        X_tr.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()

        X_tr.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        X_tr.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X_tr.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X_tr.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X_tr.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)

        X_tr.loc[segment, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        X_tr.loc[segment, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X_tr.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        

        X_tr.loc[segment, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()

        X_tr.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()

        X_tr.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()

        X_tr.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()

        X_tr.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X_tr.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X_tr.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X_tr.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        X_tr.loc[segment, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X_tr.loc[segment, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X_tr.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()



# Fill missing and infinite values

means_dict = {}

for col in X_tr.columns:

    if X_tr[col].isnull().any():

        mean_value = X_tr.loc[X_tr[col] != -np.inf, col].mean()

        X_tr.loc[X_tr[col] == -np.inf, col] = mean_value

        X_tr[col] = X_tr[col].fillna(mean_value)

        means_dict[col] = mean_value

        

print("Original shape: {}, final shape: {}".format(train.shape, X_tr.shape))

X_tr.head(3)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)



for i, seg_id in enumerate(X_test.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = pd.Series(seg['acoustic_data'].values)

    X_test.loc[seg_id, 'mean'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()

        

    X_test.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(x))

    X_test.loc[seg_id, 'mean_change_rate'] = calc_change_rate(x)

    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()

    X_test.loc[seg_id, 'std_first_50000'] = x[:50000].std()

    X_test.loc[seg_id, 'std_last_50000'] = x[-50000:].std()

    X_test.loc[seg_id, 'std_first_10000'] = x[:10000].std()

    X_test.loc[seg_id, 'std_last_10000'] = x[-10000:].std()

    

    X_test.loc[seg_id, 'avg_first_50000'] = x[:50000].mean()

    X_test.loc[seg_id, 'avg_last_50000'] = x[-50000:].mean()

    X_test.loc[seg_id, 'avg_first_10000'] = x[:10000].mean()

    X_test.loc[seg_id, 'avg_last_10000'] = x[-10000:].mean()

    

    X_test.loc[seg_id, 'min_first_50000'] = x[:50000].min()

    X_test.loc[seg_id, 'min_last_50000'] = x[-50000:].min()

    X_test.loc[seg_id, 'min_first_10000'] = x[:10000].min()

    X_test.loc[seg_id, 'min_last_10000'] = x[-10000:].min()

    

    X_test.loc[seg_id, 'max_first_50000'] = x[:50000].max()

    X_test.loc[seg_id, 'max_last_50000'] = x[-50000:].max()

    X_test.loc[seg_id, 'max_first_10000'] = x[:10000].max()

    X_test.loc[seg_id, 'max_last_10000'] = x[-10000:].max()

    

    X_test.loc[seg_id, 'max_to_min'] = x.max() / np.abs(x.min())

    X_test.loc[seg_id, 'max_to_min_diff'] = x.max() - np.abs(x.min())

    X_test.loc[seg_id, 'count_big'] = len(x[np.abs(x) > 500])

    X_test.loc[seg_id, 'sum'] = x.sum()

    

    X_test.loc[seg_id, 'mean_change_rate_first_50000'] = calc_change_rate(x[:50000])

    X_test.loc[seg_id, 'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])

    X_test.loc[seg_id, 'mean_change_rate_first_10000'] = calc_change_rate(x[:10000])

    X_test.loc[seg_id, 'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

    

    X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)

    X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)

    X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)

    X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)

    

    X_test.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(x), 0.95)

    X_test.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(x), 0.99)

    X_test.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(x), 0.05)

    X_test.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(x), 0.01)

    

    X_test.loc[seg_id, 'trend'] = add_trend_feature(x)

    X_test.loc[seg_id, 'abs_trend'] = add_trend_feature(x, abs_values=True)

    X_test.loc[seg_id, 'abs_mean'] = np.abs(x).mean()

    X_test.loc[seg_id, 'abs_std'] = np.abs(x).std()

    

    X_test.loc[seg_id, 'mad'] = x.mad()

    X_test.loc[seg_id, 'kurt'] = x.kurtosis()

    X_test.loc[seg_id, 'skew'] = x.skew()

    X_test.loc[seg_id, 'med'] = x.median()

    

    X_test.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()

    X_test.loc[seg_id, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()

    X_test.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()

    X_test.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()

    X_test.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()

    X_test.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()

    X_test.loc[seg_id, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()

    X_test.loc[seg_id, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()

    X_test.loc[seg_id, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()

    X_test.loc[seg_id, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

    X_test.loc[seg_id, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    X_test.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)

    X_test.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)

    X_test.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)

    no_of_std = 3

    X_test.loc[seg_id, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()

    X_test.loc[seg_id,'MA_700MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X_test.loc[seg_id,'MA_700MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X_test.loc[seg_id, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()

    X_test.loc[seg_id,'MA_400MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X_test.loc[seg_id,'MA_400MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X_test.loc[seg_id, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()

    X_test.drop('Moving_average_700_mean', axis=1, inplace=True)

    

    X_test.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    X_test.loc[seg_id, 'q999'] = np.quantile(x,0.999)

    X_test.loc[seg_id, 'q001'] = np.quantile(x,0.001)

    X_test.loc[seg_id, 'ave10'] = stats.trim_mean(x, 0.1)

    

    for windows in [10, 100, 1000]:

        x_roll_std = x.rolling(windows).std().dropna().values

        x_roll_mean = x.rolling(windows).mean().dropna().values

        

        X_test.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X_test.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()

        X_test.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()

        X_test.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        X_test.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X_test.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X_test.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X_test.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)

        X_test.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        X_test.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X_test.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        

        X_test.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()

        X_test.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()

        X_test.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()

        X_test.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()

        X_test.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X_test.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X_test.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X_test.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        X_test.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X_test.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X_test.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()



for col in X_test.columns:

    if X_test[col].isnull().any():

        X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]

        X_test[col] = X_test[col].fillna(means_dict[col])
class BoostingModel(object):

    def __init__(self, params, verbose=False, early_stopping=50, num_folds=10):

        self.hyperparams = params

        self.stop = early_stopping

        self.num_folds = num_folds

        self.verbose = verbose

        self.seed = RANDOM_SEED

        self.hyperparams['random_seed'] = self.seed

        self.hyperparams['objective'] = 'regression_l1'

        

    def train(self, X, y):

        """Train num_folds estimators (boosters) using KFold.

        

        Arguments:

            X: pandas dataframe with features

            y: series or dataframe with target values for regression

        """

        self.estimators = []

        self.feat_importance = pd.DataFrame({'feature': X.columns})

        self.folds = KFold(self.num_folds, shuffle=True, random_state=self.seed)

        for i, (train_index, valid_index) in enumerate(self.folds.split(X, y)):

            estimator = lgb.LGBMRegressor(**self.hyperparams)

            estimator.fit(X.iloc[train_index], y.iloc[train_index],

                          early_stopping_rounds=self.stop, verbose=self.verbose,

                          eval_set=[(X.iloc[train_index], y.iloc[train_index]),

                                    (X.iloc[valid_index], y.iloc[valid_index])])

            

            fn = "fold_" + str(i+1)

            self.feat_importance[fn] = estimator.booster_.feature_importance(importance_type='gain')

            self.estimators.append(estimator)

        self.feat_importance['fold_mean'] = self.feat_importance.mean(axis=1)

        self.feat_importance['fold_std'] = self.feat_importance.std(axis=1)

        self.feat_importance.sort_values(by='fold_mean', ascending=False, inplace=True)

        

    def predict(self, X):

        """Makes predictions for test set with each estimator and returns the average."""

        predictions = np.zeros(X.shape[0])

        for estimator in self.estimators:

            predictions += estimator.predict(X) / len(self.estimators)

        return predictions

    

    def predict_oof(self, X):

        """Returns the predictions for validation data in each fold."""

        oof_predictions = np.zeros(X.shape[0])

        for i, (train_index, valid_index) in enumerate(self.folds.split(X)):

            oof_predictions[valid_index] = self.estimators[i].predict(X.iloc[valid_index])

        return oof_predictions

    

    def validation_score(self, X, y):

        oof_predictions = self.predict_oof(X)

        return mean_absolute_error(y, oof_predictions)



params = {

    'learning_rate': 0.005,

    'num_leaves': 8,

    'max_depth': 8,

    'feature_fraction': 0.8,

    'subsample': 0.9,

    'lambda_l1': 0,

    'lambda_l2': 0.4,

    'min_data_in_leaf': 40,

    'min_gain_to_split': 0.001,

    'boosting': 'gbdt',

    'verbosity': -1,

    'n_estimators': 20000

}

model = BoostingModel(params)

model.train(X_tr, y_tr)

benchmark = model.validation_score(X_tr, y_tr)

print("Baseline score with all features (mae): {:.6f}".format(benchmark))
model_ = BoostingModel(params)

scores = []

for k in range(6, 60, 2):

    values = SelectKBest(mutual_info_regression, k=k).fit_transform(X_tr, y_tr)

    X = pd.DataFrame(values, columns=[i for i in range(values.shape[1])])

    model_.train(X, y_tr)

    scores.append(model_.validation_score(X, y_tr))



lineplot(list(range(6, 60, 2)), scores, 'FS with Mutual information')
scores = []

for n in range(6, 60, 2):

    features = model.feat_importance[:n].feature

    model_.train(X_tr[features], y_tr)

    scores.append(model_.validation_score(X_tr[features], y_tr))

    

lineplot(list(range(6, 60, 2)), scores)
diff = {}

for col in X_tr.columns:

    save = X_tr[col].values.copy()

    X_tr[col] = np.random.permutation(X_tr[col])

    diff[col] = model.validation_score(X_tr, y_tr) - benchmark

    X_tr[col] = save
scores = []

for num_feats in range(6, 60, 2):

    features = sorted(diff, key=diff.get, reverse=True)[:num_feats]

    model_.train(X_tr[features], y_tr)

    scores.append(model_.validation_score(X_tr[features], y_tr))

    

lineplot(list(range(6, 60, 2)), scores)
num_feats = scores.index(min(scores))

features = sorted(diff, key=diff.get, reverse=True)[:num_feats]

model_.train(X_tr[features], y_tr)

submission['time_to_failure'] = model_.predict(X_test[features])

submission.to_csv('submission_pi.csv')
scores = []

features = []

importance = model.feat_importance.feature.values

X = X_tr.copy()

for i in range(X_tr.shape[1] - 1):

    X.drop(importance[-1], axis=1, inplace=True)

    model_.train(X, y_tr)

    scores.append(model_.validation_score(X, y_tr))

    features.append(list(X.columns))

    importance = model_.feat_importance.feature.values



lineplot(list(range(X_tr.shape[1], 1, -1)), scores)
features = features[scores.index(min(scores))]

model_.train(X_tr[features], y_tr)

submission['time_to_failure'] = model_.predict(X_test[features])

submission.to_csv('submission_rfe.csv')
estimator = RandomForestRegressor(n_estimators=500, n_jobs=4, max_depth=4)

boruta_selector = BorutaPy(estimator, n_estimators='auto', verbose=0)

boruta_selector.fit(X_tr.values, y_tr.values.flatten())



boruta_df = pd.DataFrame({'feature': X_tr.columns, 'rank': boruta_selector.ranking_})

feats = boruta_df[boruta_df['rank'] == 1].feature

model_.train(X_tr[feats], y_tr)

print("Boruta validation score: {:.6f}".format(model_.validation_score(X_tr[feats], y_tr)))
actual_importance = model.feat_importance[['feature', 'fold_mean']]

null_importance = pd.DataFrame()

for i in range(200):

    # Get current run importances

    shuffled_target = y_tr.copy().sample(frac=1.0)

    model_.train(X_tr, shuffled_target)

    importance = model_.feat_importance[['feature', 'fold_mean']]

    importance['run'] = i + 1

    # Concat the latest importances with the old ones

    null_importance = pd.concat([null_importance, importance])

    

null_importance.head()
feature_scores = []

for _f in actual_importance['feature'].unique():

    f_null_imps_gain = null_importance.loc[null_importance['feature'] == _f, 'fold_mean'].values

    f_act_imps_gain = actual_importance.loc[actual_importance['feature'] == _f, 'fold_mean'].mean()

    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero

    feature_scores.append((_f, gain_score))



scores_df = pd.DataFrame(feature_scores, columns=['feature', 'gain_score'])

scores_df.sort_values('gain_score', ascending=False, inplace=True)



for num_feats in range(6, 60, 2):

    features = scores_df.loc[:num_feats, 'feature']

    model_.train(X_tr[features], y_tr)

    scores.append(model_.validation_score(X_tr[features], y_tr))

    

lineplot(list(range(6, 60, 2)), scores)
scores = []

for num_feats in range(6, 60, 2):

    values = SelectFromModel(LassoCV(cv=5), max_features=num_feats,

                             threshold=-np.inf).fit_transform(X_tr, y_tr)

    X = pd.DataFrame(values, columns=[i for i in range(num_feats)])

    model_.train(X, y_tr)

    scores.append(model_.validation_score(X, y_tr))

    

lineplot(list(range(6, 60, 2)), scores, "Feature selection with Lasso")