import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

import joblib



import warnings

warnings.simplefilter("ignore")
BASE_PATH = '/kaggle/input/liverpool-ion-switching/'

MODEL_PATH = '/kaggle/input/ensemble-models/'



ROW_PER_BATCH = 500000



WINDOW_SIZES = [10, 50]

RANDOM_STATE = 42
def add_test_batch_rem_drift(test):



    test['batch'] = 0



    for i in range(0, test.shape[0]//ROW_PER_BATCH):

        test.iloc[i * ROW_PER_BATCH: (i+1) * ROW_PER_BATCH,2] = i



    test['signal_undrifted'] = test.signal

    # REMOVE BATCH 1 DRIFT

    start=500

    a = 0; b = 100000

    test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.

    start=510

    a = 100000; b = 200000

    test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.

    start=540

    a = 400000; b = 500000

    test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.



    # REMOVE BATCH 2 DRIFT

    start=560

    a = 600000; b = 700000

    test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.

    start=570

    a = 700000; b = 800000

    test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.

    start=580

    a = 800000; b = 900000

    test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - 3*(test.time.values[a:b]-start)/10.



    # REMOVE BATCH 3 DRIFT

    def f(x):

        return -(0.00788)*(x-625)**2+2.345 +2.58

    a = 1000000; b = 1500000

    test.loc[test.index[a:b],'signal_undrifted'] = test.signal.values[a:b] - f(test.time[a:b].values)

    return test
def create_advance_features(df, window_sizes):

    for window in window_sizes:

        df["rolling_mean_" + str(window)] = df['signal_undrifted'].rolling(window=window).mean()

        df["rolling_std_" + str(window)] = df['signal_undrifted'].rolling(window=window).std()

        df["rolling_var_" + str(window)] = df['signal_undrifted'].rolling(window=window).var()

        df["rolling_min_" + str(window)] = df['signal_undrifted'].rolling(window=window).min()

        df["rolling_max_" + str(window)] = df['signal_undrifted'].rolling(window=window).max()

        df["rolling_min_max_ratio_" + str(window)] = df["rolling_min_" + str(window)] / df["rolling_max_" + str(window)]

        df["rolling_min_max_diff_" + str(window)] = df["rolling_max_" + str(window)] - df["rolling_min_" + str(window)]

        a = (df['signal'] - df['rolling_min_' + str(window)]) / (df['rolling_max_' + str(window)] - df['rolling_min_' + str(window)])

        df["norm_" + str(window)] = a * (np.floor(df['rolling_max_' + str(window)]) - np.ceil(df['rolling_min_' + str(window)]))



    df = df.replace([np.inf, -np.inf], np.nan)    

    df.fillna(0, inplace=True)



    return df





def features(df):

    

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    

    df['batch'] = df.index // 50_000

    df['batch_index'] = df.index  - (df.batch * 50_000)

    df['batch_slices'] = df['batch_index']  // 5_000

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal_undrifted'].mean()

        d['median'+c] = df.groupby([c])['signal_undrifted'].median()

        d['max'+c] = df.groupby([c])['signal_undrifted'].max()

        d['min'+c] = df.groupby([c])['signal_undrifted'].min()

        d['std'+c] = df.groupby([c])['signal_undrifted'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.min(np.abs(x)))

        for v in d:

            df[v] = df[c].map(d[v].to_dict())

        df['range'+c] = df['max'+c] - df['min'+c]

        df['maxtomin'+c] = df['max'+c] / df['min'+c]

        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2

    

    #add shifts

    df['signal_shift_+1'] = [0,] + list(df['signal_undrifted'].values[:-1])

    df['signal_shift_-1'] = list(df['signal_undrifted'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan

        



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal_undrifted', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:

        df[c+'_msignal'] = df[c] - df['signal_undrifted']

        

    return df
test = pd.read_csv(BASE_PATH + 'test.csv')



test = add_test_batch_rem_drift(test)

test = create_advance_features(test, WINDOW_SIZES)

test = features(test)
cols_to_remove = ['time','signal','open_channels','batch','batch_index','batch_slices','batch_slices2']

cols = [c for c in test.columns if c not in cols_to_remove]



test = test[cols]
model = joblib.load(MODEL_PATH + 'lgb_0.sav')
y_pred = model.predict(test, num_iteration=model.best_iteration)

y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)
y_pred
sub = pd.read_csv(BASE_PATH + 'sample_submission.csv')

sub['open_channels'] =  np.array(np.round(y_pred,0), np.int) 



sub.to_csv('submission.csv', index=False, float_format='%.4f')

sub.head(10)