# This is a code cell
a = 3
b = 5
a   # The last variable is always printed
a += b
list()
# First we will import all the modules we will use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# Use ! to execute a shell command from the notebook
!ls /kaggle/input/weather-postprocessing
# We can just peek at the data from the command line
!head /kaggle/input/weather-postprocessing/pp_train.csv
df_train = pd.read_csv('/kaggle/input/weather-postprocessing/pp_train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/weather-postprocessing/pp_test.csv', index_col=0)
df_train.head()
df_train.head().T    # Use .T for a transposed view
df_train.describe().T
# How big is each dataset
len(df_train), len(df_test)
# What are the columns
df_train.columns
df_test.columns
df_train.dtypes
df_train.station.nunique()
df_train.t2m_obs.hist(bins=50);
df_train[::1000].plot.scatter('t2m_obs', 't2m_fc_mean', alpha=0.5);
def linear_model(x, a, b):
    return a*x + b
x = df_train.t2m_fc_mean
y_pred = linear_model(x, a=1, b=0)
# y_pred = linear_model(x, a=1, b=0.1)
df_train[::1000].plot.scatter('t2m_obs', 't2m_fc_mean', alpha=0.5);
plt.scatter(x, y_pred, c='r', alpha=0.5);
y_true = df_train.t2m_obs
def mse(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()
mse(y_true, y_pred)
df_train.isna().sum()
df_test.isna().sum()
df_train = df_train.dropna(subset=['t2m_obs'])
# Replace missing soil moisture values with mean value
df_train.loc[:, 'sm_fc_mean'].replace(np.nan, df_train['sm_fc_mean'].mean(), inplace=True)
# Same for test dataset, using the training values
df_test.loc[:, 'sm_fc_mean'].replace(np.nan, df_train['sm_fc_mean'].mean(), inplace=True)
split_date = '2015-01-01'
X_train = df_train[df_train.time < split_date][['t2m_fc_mean']]
y_train = df_train[df_train.time < split_date]['t2m_obs']
X_valid = df_train[df_train.time >= split_date][['t2m_fc_mean']]
y_valid = df_train[df_train.time >= split_date]['t2m_obs']
X_train.shape, X_valid.shape
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_valid)
mse(y_pred, y_valid)
a = lr.coef_
b = lr.intercept_
a, b
plt.scatter(X_train[::1000], y_train[::1000], alpha=0.5)
x = np.array([-15, 20])
plt.plot(x, a*x+b, c='r');
lr.score(X_valid, y_valid)
def print_scores(model):
    r2_train = model.score(X_train, y_train)
    r2_valid = model.score(X_valid, y_valid)
    mse_train = mse(y_train, model.predict(X_train))
    mse_valid = mse(y_valid, model.predict(X_valid))
    print(f'Train R2 = {r2_train}\nValid R2 = {r2_valid}\nTrain MSE = {mse_train}\nValid MSE = {mse_valid}')
print_scores(lr)
split_date = '2015-01-01'
X_train = df_train[df_train.time < split_date].drop(['t2m_obs', 'station', 'time'], axis=1)
y_train = df_train[df_train.time < split_date]['t2m_obs']

X_valid = df_train[df_train.time >= split_date].drop(['t2m_obs', 'station', 'time'], axis=1)
y_valid = df_train[df_train.time >= split_date]['t2m_obs']

X_test  = df_test.drop(['station', 'time'], axis=1)
X_train.shape, X_test.shape
lr = LinearRegression()
lr.fit(X_train, y_train)
print_scores(lr)
preds = lr.predict(X_test)
preds.shape
sub =  pd.DataFrame({'id': range(len(preds)), 'Prediction': preds})
sub.head()
sub.to_csv('submission.csv', index=False)
X_train.to_csv('X_train.csv')
y_train.to_csv('y_train.csv')
X_valid.to_csv('X_valid.csv')
y_valid.to_csv('y_valid.csv')
X_test.to_csv('X_test.csv')
y_clim = y_train.mean()
y_clim
mse(y_valid, y_clim)
stations_train = df_train.station[df_train.time < split_date]
stations_valid = df_train.station[df_train.time >= split_date]
stations_test = df_test.station
preds = np.zeros(len(y_valid))
for s in stations_valid.unique():
    s_idxs = stations_valid == s
    if s in stations_train.unique():
        preds[s_idxs] = y_train[stations_train == s].mean()
    else:
        preds[s_idxs] = y_train.mean()
mse(preds, y_valid)
%%time
models = {}
for s in stations_train.unique():
    m = LinearRegression()
    m.fit(X_train[stations_train == s], y_train[stations_train == s])
    models[s] = m
preds = np.zeros(len(y_valid))
for s in stations_valid.unique():
    s_idxs = stations_valid == s
    if s in stations_train.unique():
        preds[s_idxs] = models[s].predict(X_valid[s_idxs])
    else:
        preds[s_idxs] = lr.predict(X_valid[s_idxs])
mse(preds, y_valid)
test_preds = np.zeros(len(X_test))
for s in stations_test.unique():
    s_idxs = stations_test == s
    if s in stations_train.unique():
        test_preds[s_idxs] = models[s].predict(X_test[s_idxs])
    else:
        test_preds[s_idxs] = lr.predict(X_test[s_idxs])
sub =  pd.DataFrame({'id': range(len(test_preds)), 'Prediction': test_preds})
sub.head()
sub.to_csv('submission_local.csv', index=False)
from sklearn.ensemble import RandomForestRegressor
%%time
rf = RandomForestRegressor(n_jobs=-1, verbose=1, min_samples_leaf=100)
rf.fit(X_train, y_train)
preds = rf.predict(X_valid)
mse(y_valid, preds)
