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