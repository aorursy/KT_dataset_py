import numpy as np 

import pandas as pd 

import sys

import gc

from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve



pd.options.display.precision = 15
train_set = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
segments = int(np.floor(train_set.shape[0] / 150000))
X_train = pd.DataFrame(index=range(segments), dtype=np.float64)

y_train = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
def feature_generate(df,x,seg):

    df.loc[seg, 'ave'] = x.mean()

    df.loc[seg, 'std'] = x.std()

    df.loc[seg, 'max'] = x.max()

    df.loc[seg, 'min'] = x.min()

    df.loc[seg, 'sum'] = x.sum()

    df.loc[seg, 'mad'] = x.mad()

    df.loc[seg, 'kurtosis'] = x.kurtosis()

    df.loc[seg, 'skew'] = x.skew()

    df.loc[seg, 'quant0_01'] = np.quantile(x,0.01)

    df.loc[seg, 'quant0_05'] = np.quantile(x,0.05)

    df.loc[seg, 'quant0_95'] = np.quantile(x,0.95)

    df.loc[seg, 'quant0_99'] = np.quantile(x,0.99)

    df.loc[seg, 'abs_min'] = np.abs(x).min()

    df.loc[seg, 'abs_max'] = np.abs(x).max()

    df.loc[seg, 'abs_mean'] = np.abs(x).mean()

    df.loc[seg, 'abs_std'] = np.abs(x).std()

    df.loc[seg, 'mean_change_abs'] = np.mean(np.diff(x))

    df.loc[seg, 'max_to_min'] = x.max() / np.abs(x.min())

    df.loc[seg, 'max_to_min_diff'] = x.max() - np.abs(x.min())

    df.loc[seg, 'count_big'] = len(x[np.abs(x) > 500])

    df.loc[seg, 'average_first_10000'] = x[:10000].mean()

    df.loc[seg, 'average_last_10000']  =  x[-10000:].mean()

    df.loc[seg, 'average_first_50000'] = x[:50000].mean()

    df.loc[seg, 'average_last_50000'] = x[-50000:].mean()

    df.loc[seg, 'std_first_10000'] = x[:10000].std()

    df.loc[seg, 'std_last_10000']  =  x[-10000:].std()

    df.loc[seg, 'std_first_50000'] = x[:50000].std()

    df.loc[seg, 'std_last_50000'] = x[-50000:].std()

    df.loc[seg, '10q'] = np.percentile(x, 0.10)

    df.loc[seg, '25q'] = np.percentile(x, 0.25)

    df.loc[seg, '50q'] = np.percentile(x, 0.50)

    df.loc[seg, '75q'] = np.percentile(x, 0.75)

    df.loc[seg, '90q'] = np.percentile(x, 0.90)

    df.loc[seg, 'abs_1q'] = np.percentile(x, np.abs(0.01))

    df.loc[seg, 'abs_5q'] = np.percentile(x, np.abs(0.05))

    df.loc[seg, 'abs_30q'] = np.percentile(x, np.abs(0.30))

    df.loc[seg, 'abs_60q'] = np.percentile(x, np.abs(0.60))

    df.loc[seg, 'abs_95q'] = np.percentile(x, np.abs(0.95))

    df.loc[seg, 'abs_99q'] = np.percentile(x, np.abs(0.99))

    df.loc[seg, 'hilbert_mean'] = np.abs(hilbert(x)).mean()

    df.loc[seg, 'hann_window_mean'] = (convolve(x, hann(150), mode = 'same') / sum(hann(150))).mean()    



    for windows in [10, 100, 1000]:

        x_roll_std = x.rolling(windows).std().dropna().values

        x_roll_mean = x.rolling(windows).mean().dropna().values

        df.loc[seg, 'avg_roll_std' + str(windows)] = x_roll_std.mean()

        df.loc[seg, 'std_roll_std' + str(windows)] = x_roll_std.std()

        df.loc[seg, 'max_roll_std' + str(windows)] = x_roll_std.max()

        df.loc[seg, 'min_roll_std' + str(windows)] = x_roll_std.min()

        df.loc[seg, '1q_roll_std' + str(windows)] = np.quantile(x_roll_std, 0.01)

        df.loc[seg, '5q_roll_std' + str(windows)] = np.quantile(x_roll_std, 0.05)

        df.loc[seg, '95q_roll_std' + str(windows)] = np.quantile(x_roll_std, 0.95)

        df.loc[seg, '99q_roll_std' + str(windows)] = np.quantile(x_roll_std, 0.99)

        df.loc[seg, 'av_change_abs_roll_std' + str(windows)] = np.mean(np.diff(x_roll_std))

        df.loc[seg, 'abs_max_roll_std' + str(windows)] = np.abs(x_roll_std).max()

        df.loc[seg, 'avg_roll_mean' + str(windows)] = x_roll_mean.mean()

        df.loc[seg, 'std_roll_mean' + str(windows)] = x_roll_mean.std()

        df.loc[seg, 'max_roll_mean' + str(windows)] = x_roll_mean.max()

        df.loc[seg, 'min_roll_mean' + str(windows)] = x_roll_mean.min()

        df.loc[seg, '1q_roll_mean' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        df.loc[seg, '5q_roll_mean' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        df.loc[seg, '95q_roll_mean' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        df.loc[seg, '99q_roll_mean' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        df.loc[seg, 'av_change_abs_roll_mean' + str(windows)] = np.mean(np.diff(x_roll_mean))

        df.loc[seg, 'abs_max_roll_mean' + str(windows)] = np.abs(x_roll_mean).max()   

    return df

for s in range(segments):

    seg = train_set.iloc[s*150000:s*150000+150000]

    x = pd.Series(seg['acoustic_data'].values)

    y = seg['time_to_failure'].values[-1]

    y_train.loc[s, 'time_to_failure'] = y

    X_train = feature_generate(X_train,x,s)

columns=X_train.columns  

del train_set

gc.collect()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

y_train = y_train.values.flatten()

gc.collect()
import xgboost as xgb

model = xgb.XGBRegressor(objective = 'reg:linear',

                         metric = 'mae',

                         tree_method = 'gpu_hist',

                         verbosity = 0)
%%time

model.fit(X_train,y_train)
from matplotlib import pyplot

print(model.feature_importances_)

pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)

pyplot.show()
from xgboost import plot_importance

plot_importance(model)

pyplot.show()
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(columns=columns, dtype=np.float64, index=submission.index)

for s in X_test.index:

    seg = pd.read_csv('../input/test/' + s + '.csv')

    x = pd.Series(seg['acoustic_data'].values)

    X_test = feature_generate(X_test,x,s)

X_test = scaler.transform(X_test)

submission['time_to_failure'] = model.predict(X_test).clip(0, 16)

submission.to_csv('submission.csv')