import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set_style('whitegrid')
sns.set_context('notebook')
# The pre-processed data is now stored here
!ls /kaggle/input/nb1-linear-regression
df_train = pd.read_csv('/kaggle/input/weather-postprocessing/pp_train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/weather-postprocessing/pp_test.csv', index_col=0)

X_train = pd.read_csv('/kaggle/input/nb1-linear-regression/X_train.csv', index_col=0)
y_train = pd.read_csv('/kaggle/input/nb1-linear-regression/y_train.csv', index_col=0, squeeze=True)
X_valid = pd.read_csv('/kaggle/input/nb1-linear-regression/X_valid.csv', index_col=0)
y_valid = pd.read_csv('/kaggle/input/nb1-linear-regression/y_valid.csv', index_col=0, squeeze=True)
X_test = pd.read_csv('/kaggle/input/nb1-linear-regression/X_test.csv', index_col=0)
def mse(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

def print_scores(model):
    r2_train = model.score(X_train, y_train)
    r2_valid = model.score(X_valid, y_valid)
    mse_train = mse(y_train, model.predict(X_train))
    mse_valid = mse(y_valid, model.predict(X_valid))
    print(f'Train R2 = {r2_train}\nValid R2 = {r2_valid}\nTrain MSE = {mse_train}\nValid MSE = {mse_valid}')
sns.pairplot(
    df_train[::1000], 
    x_vars=['t2m_fc_mean', 'orog', 'gh_pl500_fc_mean', 'cape_fc_mean', 'ssr_fc_mean', 'sm_fc_mean', 'u10_fc_mean'], 
    y_vars=['t2m_obs']
);
from sklearn.tree import DecisionTreeRegressor, plot_tree
dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt, filled=True, ax=ax, fontsize=12, feature_names=X_train.columns);
y_train.mean()
dt = DecisionTreeRegressor()
%time dt.fit(X_train, y_train)
print_scores(dt)
dt = DecisionTreeRegressor(min_samples_leaf=200)
%time dt.fit(X_train, y_train)
print_scores(dt)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, n_jobs=-1)
%time rf.fit(X_train, y_train)
print_scores(rf)
rf = RandomForestRegressor(n_estimators=10, n_jobs=-1, min_samples_leaf=100)
%time rf.fit(X_train, y_train)
print_scores(rf)
rf = RandomForestRegressor(n_estimators=40, n_jobs=-1, min_samples_leaf=20)
%time rf.fit(X_train, y_train)
print_scores(rf)
dt1 = rf.estimators_[0]
print_scores(dt1)
plt.figure(figsize=(10, 7))
plt.barh(X_train.columns, rf.feature_importances_)
plt.xscale('log')
plt.tight_layout()
X_valid.shape
X_pdp = X_valid.copy()
X_pdp['station_alt'] = 100
rf.predict(X_pdp).mean()
X_pdp['station_alt'] = 500
rf.predict(X_pdp).mean()
from sklearn.inspection import plot_partial_dependence
order = np.argsort(rf.feature_importances_)[::-1]
fig, ax = plt.subplots(figsize=(18, 3))
plot_partial_dependence(rf, X_valid[::1000], order[1:8], feature_names=X_train.columns, grid_resolution=5, n_jobs=-1, n_cols=7, ax=ax)
plt.tight_layout()
preds = rf.predict(X_test)
sub =  pd.DataFrame({'id': range(len(preds)), 'Prediction': preds})
sub.to_csv('submission.csv', index=False)
df_train = df_train.dropna(subset=['t2m_obs'])
split_date = '2015-01-01'
stations_train = df_train.station[df_train.time < split_date]
stations_valid = df_train.station[df_train.time >= split_date]
stations_test = df_test.station
from tqdm.notebook import tqdm
models = {}
for s in tqdm(stations_train.unique()):
    m = RandomForestRegressor(n_estimators=40, n_jobs=-1, min_samples_leaf=10)
    m.fit(X_train[stations_train == s], y_train[stations_train == s])
    models[s] = m
preds = np.zeros(len(y_valid))
for s in stations_valid.unique():
    s_idxs = stations_valid == s
    if s in stations_train.unique():
        preds[s_idxs] = models[s].predict(X_valid[s_idxs])
    else:
        preds[s_idxs] = rf.predict(X_valid[s_idxs])
mse(preds, y_valid)
test_preds = np.zeros(len(X_test))
for s in stations_test.unique():
    s_idxs = stations_test == s
    if s in stations_train.unique():
        test_preds[s_idxs] = models[s].predict(X_test[s_idxs])
    else:
        test_preds[s_idxs] = rf.predict(X_test[s_idxs])
sub =  pd.DataFrame({'id': range(len(test_preds)), 'Prediction': test_preds})
sub.head()
sub.to_csv('submission_local.csv', index=False)
unique_stations = pd.concat([df_train.station, df_test.station]).unique()
stat2id = {s: i for i, s in enumerate(unique_stations)}
df_train['station'] = df_train.station.apply(lambda x: stat2id[x])
df_test['station'] = df_test.station.apply(lambda x: stat2id[x])
# Replace missing soil moisture values with mean value
df_train.loc[:, 'sm_fc_mean'].replace(np.nan, df_train['sm_fc_mean'].mean(), inplace=True)
# Same for test dataset, using the training values
df_test.loc[:, 'sm_fc_mean'].replace(np.nan, df_train['sm_fc_mean'].mean(), inplace=True)
X_train = df_train[df_train.time < split_date].drop(['t2m_obs', 'time'], axis=1)
y_train = df_train[df_train.time < split_date]['t2m_obs']

X_valid = df_train[df_train.time >= split_date].drop(['t2m_obs', 'time'], axis=1)
y_valid = df_train[df_train.time >= split_date]['t2m_obs']

X_test  = df_test.drop(['time'], axis=1)
rf = RandomForestRegressor(n_estimators=40, n_jobs=-1, min_samples_leaf=20)
%time rf.fit(X_train, y_train)
print_scores(rf)
