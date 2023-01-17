# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

from sklearn import *

import lightgbm as lgb

pd.set_option('display.max_rows', 1000)  # or 1000



train = pd.read_csv('/kaggle/input/ionswitchingkl/datasets/trainK.csv')
pd.set_option('display.max_columns', 1000)  # or 1000

pd.set_option('display.max_rows', 1000)  # or 1000

pd.set_option('display.max_colwidth', 199)  # or 199

import warnings

warnings.filterwarnings("ignore")
def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    df['batch_slices'] = df['batch_index']  // 2500

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)



    print('Pre-Processing stat features....')

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal'].mean()

        d['var'+c] = df.groupby([c])['signal'].var()

        d['max'+c] = df.groupby([c])['signal'].max()

        d['min'+c] = df.groupby([c])['signal'].min()

        d['std'+c] = df.groupby([c])['signal'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        for v in d:

            df[v] = df[c].map(d[v].to_dict())

        df['range'+c] = df['max'+c] - df['min'+c]

        df['maxtomin'+c] = df['max'+c] / df['min'+c]

        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2

    

    print('Pre-Processing shifting features....')

    

    df['shift+1_'] = df['signal'].shift(1)

    df['shift-1_'] = df['signal'].shift(-1)

    df['shift+2_'] = df['signal'].shift(2)

    df['shift-2_'] = df['signal'].shift(-2)

    df['shift+3_'] = df['signal'].shift(3)

    df['shift-3_'] = df['signal'].shift(-3)

    df['shift+4_'] = df['signal'].shift(4)

    df['shift-4_'] = df['signal'].shift(-4)

    df['shift+5_'] = df['signal'].shift(5)

    df['shift-5_'] = df['signal'].shift(-5)

        

    

    for c in [20,50,100,500]:

        print('Pre-Processing rolling window '+str(c)+'.....')

        df['rolling_mean'+str(c)] = df['signal'].rolling(c).mean().tolist()

        df['rolling_std'+str(c)] = df['signal'].rolling(c).std().tolist()

        df['rolling_var'+str(c)] = df['signal'].rolling(c).var().tolist()

        

    df = df.fillna(method='bfill').fillna(method='ffill')



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    return df



training = features(train)
del train
training_columns = training.columns

col = [c for c in training_columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]

Y = training['open_channels'].values
X = training[col].values

del training
import gc

gc.collect()
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=7)

del X

del Y
def MacroF1Metric(preds, dtrain):

    from sklearn.metrics import f1_score

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = f1_score(labels, preds, average = 'macro')

    return ('MacroF1Metric', score, True)



params = {}

params['application']='root_mean_squared_error'

params['num_boost_round'] = 3000

params['learning_rate'] = 0.01

params['boosting_type'] = 'gbdt'

params['metric'] = 'rmse'

params['sub_feature'] = 0.833

params['num_leaves'] = 207

params['min_split_gain'] = 0.05

params['min_child_weight'] = 27

params['max_depth'] = -1

params['num_threads'] = 20

params['max_bin'] = 50

params['lambda_l2'] = 0.10

params['lambda_l1'] = 0.30

params['feature_fraction']= 0.833

params['bagging_fraction']= 0.979

params['seed']=1729

params['device_type'] = 'gpu'

model = lgb.train(params, lgb.Dataset(X_train, Y_train), 2000, lgb.Dataset(X_test, Y_test),verbose_eval=100, early_stopping_rounds=200, feval=MacroF1Metric)
del X_train

del X_test

del Y_train

del Y_test
import gc

gc.collect()
test = pd.read_csv('/kaggle/input/ionswitchingkl/datasets/testK.csv')

test = features(test)
preds = model.predict(test[col], num_iteration=model.best_iteration)

test['open_channels'] = np.round(np.clip(preds, 0, 10)).astype(int)

test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')
import matplotlib.pyplot as plt

fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

lgb.plot_importance(model,ax = axes,height = 0.5)

plt.show();plt.close()

import gc

gc.collect()