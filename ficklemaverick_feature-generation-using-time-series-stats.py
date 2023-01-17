# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from scipy import stats

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

import time
trainData = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
pd.options.display.precision = 15
def calRCS(data):

    N = len(data)

    m = np.mean(data)

    sigma = np.std(data)

    s = np.cumsum(data-m) * 1.0 / (N * sigma)

    R = np.max(s) - np.min(s)

    return R
rows = 100_000

segments = int(np.floor(trainData.shape[0] / rows))



X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min','q95','q99', 'q05','q01',

                               'abs_max', 'abs_mean', 'abs_std', 'iqr', 

                                'q999','q001','ave10'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



for segment in tqdm(range(segments)):

    seg = trainData.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = x.mean()

    X_train.loc[segment, 'std'] = x.std()

    X_train.loc[segment, 'max'] = x.max()

    X_train.loc[segment, 'min'] = x.min()

    X_train.loc[segment, 'q95'] = np.quantile(x,0.95)

    X_train.loc[segment, 'q99'] = np.quantile(x,0.99)

    X_train.loc[segment, 'q05'] = np.quantile(x,0.05)

    X_train.loc[segment, 'q01'] = np.quantile(x,0.01)

    

    X_train.loc[segment, 'abs_max'] = np.abs(x).max()

    X_train.loc[segment, 'abs_mean'] = np.abs(x).mean()

    X_train.loc[segment, 'abs_std'] = np.abs(x).std()  

    X_train.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    X_train.loc[segment, 'q999'] = np.quantile(x,0.999)

    X_train.loc[segment, 'q001'] = np.quantile(x,0.001)

    X_train.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)

    X_train.loc[segment, 'skew'] = stats.skew(x)

    X_train.loc[segment, 'av_change_abs'] = np.mean(np.diff(x))

    X_train.loc[segment, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])



# Extra calculated time series features

    X_train.loc[segment, 'rcs'] = calRCS(x)

    X_train.loc[segment, 'amplitude'] = 0.5*((np.max(x))-(np.min(x)))

    X_train.loc[segment, 'meanvar'] = x.std()/x.mean()

    X_train.loc[segment , 'medianAbsDev'] = np.median(abs(x-np.median(x)))

    X_train.loc[segment , 'percentAmp'] = np.max(np.abs(x-np.median(x)))/np.median(x)

    X_train.loc[segment, 'andersonDarling'] = 1 / (1.0 + np.exp(-10 * (stats.anderson(x)[0] - 0.3)))
from sklearn.model_selection import train_test_split



Xtrain, Xtest, ytrain, ytest = train_test_split(X_train,y_train, test_size=0.2)
X_train.head()
scaler = StandardScaler()

scaler.fit(Xtrain)

Xtrain_scaled = scaler.transform(Xtrain)

Xtest_scaled = scaler.transform(Xtest)
Xtrain_scaled.shape
lgb_train = lgb.Dataset(Xtrain_scaled, ytrain.values.flatten())

lgb_eval = lgb.Dataset(Xtest_scaled, ytest, reference=lgb_train)
# specify your configurations as a dict

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'max_bin': 500,

    'metric': {'l2', 'l1'},

    'num_leaves': 200,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}
start = time.time()

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=300,valid_sets=lgb_eval,

                early_stopping_rounds=5)

end = time.time()

fit_time = (end - start)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
X_test.index
for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    

    x = seg['acoustic_data'].values

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()

    X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)

    X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)

    X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)

    X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)

    

    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()

    X_test.loc[seg_id, 'abs_mean'] = np.abs(x).mean()

    X_test.loc[seg_id, 'abs_std'] = np.abs(x).std()

    

    X_test.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    X_test.loc[seg_id, 'q999'] = np.quantile(x,0.999)

    X_test.loc[seg_id, 'q001'] = np.quantile(x,0.001)

    X_test.loc[seg_id, 'ave10'] = stats.trim_mean(x, 0.1)

    

    X_test.loc[seg_id, 'skew'] = stats.skew(x)

    X_test.loc[seg_id, 'av_change_abs'] = np.mean(np.diff(x))

    X_test.loc[seg_id, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])



# Extra calculated time series features

    X_test.loc[seg_id, 'rcs'] = calRCS(x)

    X_test.loc[seg_id, 'amplitude'] = 0.5*((np.max(x))-(np.min(x)))

    X_test.loc[seg_id, 'meanvar'] = x.std()/x.mean()

    X_test.loc[seg_id , 'medianAbsDev'] = np.median(abs(x-np.median(x)))

    X_test.loc[seg_id , 'percentAmp'] = np.max(np.abs(x-np.median(x)))/np.median(x)

    X_test.loc[seg_id, 'andersonDarling'] = 1 / (1.0 + np.exp(-10 * (stats.anderson(x)[0] - 0.3)))
X_test
X_test_scaled = scaler.transform(X_test)
X_test_scaled
submission['time_to_failure'] = gbm.predict(X_test_scaled)
submission.to_csv('submission.csv')