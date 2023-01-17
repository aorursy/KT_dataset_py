# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%%time
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.head(5)
train.shape
partial_train = train[::20]
partial_train.head(5)
partial_train['time_to_failure'].shape
figure, axes1 = plt.subplots(figsize=(18,10))

plt.title("Seismic Data Trends with 5% sample of original data")

plt.plot(partial_train['acoustic_data'], color='r')
axes1.set_ylabel('Acoustic Data', color='r')
plt.legend(['Acoustic Data'])

axes2 = axes1.twinx()
plt.plot(partial_train['time_to_failure'], color='g')
axes2.set_ylabel('Time to Failure', color='g')
plt.legend(['Time to Failure'])
# list of features to be engineered

features = ['mean','max','variance','min', 'stdev', 'max-min-diff',
            'max-mean-diff', 'mean-change-abs', 'abs-max', 'abs-min',
            'std-first-50000', 'std-last-50000', 'mean-first-50000',
            'mean-last-50000', 'max-first-50000', 'max-last-50000',
            'min-first-50000', 'min-last-50000']
# Feature Engineering

rows = 150000
segments = int(np.floor(train.shape[0] / rows))

X = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=features)
Y = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

for segment in range(segments):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    Y.loc[segment, 'time_to_failure'] = y
    
    X.loc[segment, 'mean'] = x.mean()
    X.loc[segment, 'stdev'] = x.std()
    X.loc[segment, 'variance'] = np.var(x)
    X.loc[segment, 'max'] = x.max()
    X.loc[segment, 'min'] = x.min()
    X.loc[segment, 'max-min-diff'] = x.max()-x.min()
    X.loc[segment, 'max-mean-diff'] = x.max()-x.mean()
    X.loc[segment, 'mean-change-abs'] = np.mean(np.diff(x))
    X.loc[segment, 'abs-min'] = np.abs(x).min()
    X.loc[segment, 'abs-max'] = np.abs(x).max()
    X.loc[segment, 'std-first-50000'] = x[:50000].std()
    X.loc[segment, 'std-last-50000'] = x[-50000:].std()
    X.loc[segment, 'mean-first-50000'] = x[:50000].min()
    X.loc[segment, 'mean-last-50000'] = x[-50000:].mean()
    X.loc[segment, 'max-first-50000'] = x[:50000].max()
    X.loc[segment, 'max-last-50000'] = x[-50000:].max()
    X.loc[segment, 'min-first-50000'] = x[:50000].min()
    X.loc[segment, 'min-last-50000'] = x[-50000:].min()
X.head(5)
data = pd.concat([X,Y],axis=1)
sns.set(rc={'figure.figsize': (18,12)})
sns.pairplot(data)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
X_train,X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=1210)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

scaler.fit(X_test)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
lr = LinearRegression()
lr.fit(X_train_sc,y_train)
pred = lr.predict(X_test_sc)
mean_absolute_error(y_test, pred)
from lightgbm import LGBMRegressor
params = {'num_leaves': 54,'min_data_in_leaf': 79,'objective': 'huber',
         'max_depth': -1, 'learning_rate': 0.01, "boosting": "gbdt",
         # "feature_fraction": 0.8354507676881442,
         "bagging_freq": 3,"bagging_fraction": 0.8126672064208567,
         "bagging_seed": 11,"metric": 'mae',
         "verbosity": -1,'reg_alpha': 1.1302650970728192,
         'reg_lambda': 0.3603427518866501}
lgbm = LGBMRegressor(nthread=4,n_estimators=10000,
            learning_rate=0.01,num_leaves=54,
            colsample_bytree=0.9497036,subsample=0.8715623,
            max_depth=8,reg_alpha=0.04,
            reg_lambda=0.073,min_child_weight=40,silent=-1,verbose=-1,)
lgbm.fit(X_train, y_train)
pred_lgbm = lgbm.predict(X_test)
mean_absolute_error(y_test,pred_lgbm)