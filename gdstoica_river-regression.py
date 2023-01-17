# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file = os.path.join(dirname, filename)

        if 'train' in file:

            trainset = file

        else:

            testset = file



print(trainset)



# Any results you write to the current directory are saved as output.
ds_train = pd.read_csv(trainset)

ds_train.head()
ds_train.rename(columns={"1": "ssc_st1", "2": "ssc_st2", "3":"ssc_st3", "4": "ssc_st4", "5": "ssc_st5", "6":"ssc_st6", "7": "ssc_st7"}, inplace=True)

ds_train.head()
ds_train.describe()
ds_train.isna().sum()
ds_train_s = ds_train.drop(['ssc_st3', 'ssc_st4', 'ssc_st5', 'ssc_st6', 'ssc_st7'], axis=1)

f, ax = plt.subplots(figsize=(10, 8))

corr = ds_train_s.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),

            square=True, ax=ax)

from sklearn.model_selection import train_test_split

y = ds_train_s['target']

ds_train_s.drop(['target'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

X_t, X_v, y_t, y_v = train_test_split(ds_train_s, y, test_size=0.2, random_state=72)
import seaborn as sns

sns.scatterplot(x='ssc_st1', y='ssc_st2', data=X_t)

plt.scatter(X_t['ssc_st1'], y_t), plt.scatter(X_t['ssc_st2'], y_t)
X_t['ssc_st1'].fillna((X_t['ssc_st1'].mean()), inplace=True)

X_t['ssc_st2'].fillna((X_t['ssc_st2'].mean()), inplace=True)
X_t.describe()
y_t.describe()
f= plt.figure(figsize=(12,5))



ax=f.add_subplot(121)

sns.distplot(X_t['ssc_st1'],color='c',ax=ax)



ax=f.add_subplot(122)

sns.distplot(X_t['ssc_st2'],color='r',ax=ax)
X_t.ssc_st1.loc[X_t.ssc_st1 > 40]  = 40

X_t.ssc_st2.loc[X_t.ssc_st2 > 40]  = 40
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



lr = LinearRegression().fit(X_t,y_t)

print(lr.score(X_v,y_v))
X_v.isnull().sum()
lr_train_pred = lr.predict(X_t)

lr_val_pred = lr.predict(X_v)



print('MSE train data: %.3f, MSE test data: %.3f' % (

mean_squared_error(y_t,lr_train_pred),

mean_squared_error(y_v,lr_val_pred)))



print('R2 train data: %.3f, R2 test data: %.3f' % (

r2_score(y_t,lr_train_pred),

r2_score(y_v,lr_val_pred)))
y_v.head()
lr_val_pred[:5]
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,mean_squared_error



forest = RandomForestRegressor(n_estimators = 100,

                              criterion = 'mse',

                              random_state = 72,

                              n_jobs = -1)

forest.fit(X_t,y_t)

forest_train_pred = forest.predict(X_t)

forest_val_pred = forest.predict(X_v)



print('MSE train data: %.3f, MSE test data: %.3f' % (

mean_squared_error(y_t,forest_train_pred),

mean_squared_error(y_v,forest_val_pred)))



print('R2 train data: %.3f, R2 test data: %.3f' % (

r2_score(y_t,forest_train_pred),

r2_score(y_v,forest_val_pred)))
forest_val_pred[:5]
from sklearn.ensemble import GradientBoostingRegressor



gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=10, learning_rate=0.5, random_state=72)

gbrt.fit(X_t, y_t)

gbrt_train_pred = gbrt.predict(X_t)

gbrt_val_pred = gbrt.predict(X_v)



print('MSE train data: %.3f, MSE test data: %.3f' % (

mean_squared_error(y_t,gbrt_train_pred),

mean_squared_error(y_v,gbrt_val_pred)))



print('R2 train data: %.3f, R2 test data: %.3f' % (

r2_score(y_t,gbrt_train_pred),

r2_score(y_v,gbrt_val_pred)))

gbrt_train_pred[:5]
from sklearn.svm import LinearSVR



svm_reg = LinearSVR(epsilon=0.7, random_state=72)

svm_reg.fit(X_t, y_t)



svr_train_pred = svm_reg.predict(X_t)

svr_val_pred = svm_reg.predict(X_v)



print('MSE train data: %.3f, MSE test data: %.3f' % (

mean_squared_error(y_t,svr_train_pred),

mean_squared_error(y_v,svr_val_pred)))



print('R2 train data: %.3f, R2 test data: %.3f' % (

r2_score(y_t,svr_train_pred),

r2_score(y_v,svr_val_pred)))
from sklearn.model_selection import GridSearchCV

lsvr = LinearSVR()

param_grid = { 'C':[0.05, 0.075], 'epsilon':[1.25, 1.3, 1.4, 1.5, 1.45, 1.55], 'random_state':[72]}

grid = GridSearchCV(lsvr, param_grid, n_jobs=-1)

grid.fit(X_t, y_t)



grid_train_pred = grid.predict(X_t)

grid_val_pred = grid.predict(X_v)



print('MSE train data: %.3f, MSE test data: %.3f' % (

mean_squared_error(y_t,grid_train_pred),

mean_squared_error(y_v,grid_val_pred)))



print('R2 train data: %.3f, R2 test data: %.3f' % (

r2_score(y_t,grid_train_pred),

r2_score(y_v,grid_val_pred)))



grid.best_params_
y_v.head()
y_vn = np.asarray(y_v)

y_vn
pl = pd.DataFrame(np.column_stack((y_v,lr_val_pred,forest_val_pred, gbrt_val_pred, grid_val_pred)))

pl.rename(columns={0:'target', 1:'LR', 2:'RF', 3:'GBR', 4:'SVR'}, inplace=True)

pl.plot.bar(figsize=(25,8))
pl.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_ts = sc.fit_transform(X_t)

X_vs = sc.fit_transform(X_v)



from sklearn.linear_model import Ridge

clf = Ridge(alpha=0.5, random_state=72)

clf.fit(X_ts, y_t)
clf.score(X_vs, y_v)
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.01, random_state=72)

lasso_reg.fit(X_ts, y_t)

lasso_reg.score(X_vs, y_v)
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=-0.9, l1_ratio=1, fit_intercept=True, precompute=True, tol=0.5)

elastic_net.fit(X_ts, y_t)

elastic_net.score(X_vs, y_v)