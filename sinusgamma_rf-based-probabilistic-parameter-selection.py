import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

%config InlineBackend.figure_format = 'svg'

from pylab import rcParams

rcParams['figure.figsize'] = 5, 4

import warnings

warnings.simplefilter('ignore')
import os

print(os.listdir("../input"))
df_full = pd.read_csv("../input/forex-strategy-results-first/kaggle_USD_United_States_Consumer_Price_Index_Ex_Food__Energy_YoY_USDJPY.csv")

df_full.columns
df = df_full[['enabledOrderDirection', 'pointsAway', 'stopLoss', 'trailingStop', 'maxSlippage', 'finishDep']].copy()

df.sort_values(by='finishDep', ascending=False).head()
df.loc[df['enabledOrderDirection'] == 'LONG', 'enabledOrderDirection'] = 1

df.loc[df['enabledOrderDirection'] == 'SHORT', 'enabledOrderDirection'] = -1

df.head()
print(df.shape)

print(df.info())
df.describe()
df.loc[df['finishDep'] > 1000, 'finishDep'].count() / df.shape[0]
df.loc[df['finishDep'] < 1000, 'finishDep'].count() / df.shape[0]
df.loc[df['finishDep'] == 1000, 'finishDep'].count() / df.shape[0]
df['isInProfit'] = df['finishDep'].map(lambda x: (x > 1000))

df.head()
pd.crosstab(df['enabledOrderDirection'], df['isInProfit'], margins=True)
sns.countplot(x='enabledOrderDirection', hue='isInProfit', data=df);
fig, ax = plt.subplots(figsize=(14,5))

sns.distplot(df[df['enabledOrderDirection']==-1]['finishDep'], ax=ax, color='r', bins=range(0, 1500, 10), hist_kws=dict(alpha=0.5))

sns.distplot(df[df['enabledOrderDirection']==1]['finishDep'], ax=ax, color='g', bins=range(0, 1500, 10), hist_kws=dict(alpha=0.5))

ax.axvline(1000, color='k', linestyle='--')

ax.set_xlim(700, 1300);
plt.scatter(df['pointsAway'], df['trailingStop'])

plt.xlabel('pointsAway')

plt.ylabel('trailingStop');
heatmapShort_data = pd.pivot_table(df[df['enabledOrderDirection']==-1], values='finishDep', 

                     index=['pointsAway'], 

                     columns='trailingStop')

heatmapLong_data = pd.pivot_table(df[df['enabledOrderDirection']==1], values='finishDep', 

                     index=['pointsAway'], 

                     columns='trailingStop')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,6))

sns.heatmap(heatmapShort_data, ax=ax1, vmin=700, vmax=1300, cmap="seismic")

sns.heatmap(heatmapLong_data, ax=ax2, vmin=700, vmax=1300, cmap="seismic")
heatmapShort_data = pd.pivot_table(df[df['enabledOrderDirection']==-1], values='isInProfit', 

                     index=['pointsAway'], 

                     columns='trailingStop')

heatmapLong_data = pd.pivot_table(df[df['enabledOrderDirection']==1], values='isInProfit', 

                     index=['pointsAway'], 

                     columns='trailingStop')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,6))

sns.heatmap(heatmapShort_data, ax=ax1, cmap="seismic")

sns.heatmap(heatmapLong_data, ax=ax2, cmap="seismic")
heatmapSL_data = pd.pivot_table(df[df['enabledOrderDirection']==1], values='isInProfit', 

                     index=['pointsAway'], 

                     columns='stopLoss')

heatmapMS_data = pd.pivot_table(df[df['enabledOrderDirection']==1], values='isInProfit', 

                     index=['pointsAway'], 

                     columns='maxSlippage')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,6))

sns.heatmap(heatmapSL_data, ax=ax1, cmap="seismic")

sns.heatmap(heatmapMS_data, ax=ax2, cmap="seismic")
swarms = ['pointsAway', 'trailingStop', 'stopLoss', 'maxSlippage']

for hue in swarms:

    sns.catplot(x="enabledOrderDirection", y="finishDep", hue=hue, data=df, 

            size=5, aspect=2, palette="inferno", s=4, kind='swarm')

df_long = df.loc[df['enabledOrderDirection'] == 1]

print(df_long.shape)

df_long.head()
df_full = pd.read_csv("../input/forex-strategy-results-next/kaggleB_USD_United_States_Consumer_Price_Index_Ex_Food__Energy_YoY_USDJPY.csv")

df_full.columns
df_full = df_full.loc[df_full['enabledOrderDirection'] =='LONG']

df_full.shape
df = df_full[['pointsAway', 'stopLoss', 'trailingStop', 'breakevenTrigger', 'breakevenDistance', 'maxSlippage', 'finishDep', 'profitNb', 'orderPercent', 'profitPercent', 'avrPLclosedorder', 'PLrateCom', 'profitNbWithComm']].copy()

df['isInProfit'] = df['finishDep'].map(lambda x: (x > 1000))

df.sort_values(by='finishDep', ascending=False).head()
plt.scatter(df['pointsAway'], df['finishDep'])

plt.xlabel('pointsAway')

plt.ylabel('finishDep');
heatmapTS_data = pd.pivot_table(df, values='isInProfit', 

                     index=['pointsAway'], 

                     columns='trailingStop')

heatmapBT_data = pd.pivot_table(df, values='isInProfit', 

                     index=['pointsAway'], 

                     columns='breakevenTrigger')

heatmapBD_data = pd.pivot_table(df, values='isInProfit', 

                     index=['pointsAway'], 

                     columns='breakevenDistance')

heatmapFD_data = pd.pivot_table(df, values='finishDep', 

                     index=['pointsAway'], 

                     columns='breakevenDistance')

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(16,4))

sns.heatmap(heatmapTS_data, ax=ax1, cmap="seismic")

sns.heatmap(heatmapBT_data, ax=ax2, cmap="seismic")

sns.heatmap(heatmapBD_data, ax=ax3, cmap="seismic")

sns.heatmap(heatmapFD_data, ax=ax4, cmap="seismic", center=1000)
dfs=df.sample(n=500, replace=False, random_state=1)

sns.swarmplot(y=dfs["finishDep"], hue=dfs["breakevenTrigger"], x=[""]*len(dfs), size=5, palette="inferno", s=4)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import GridSearchCV, StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 

                            class_weight='balanced')

parameters = {'max_features': [2, 4, 6], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}



X = df[['pointsAway', 'stopLoss', 'trailingStop', 'breakevenTrigger', 'breakevenDistance', 'maxSlippage']]

y_IIP = df['isInProfit']
%%time

rf_IIP = GridSearchCV(rf, parameters, n_jobs=-1, scoring='roc_auc', cv=skf, verbose=True)

rf_IIP = rf_IIP.fit(X, y_IIP)

print(rf_IIP.best_score_)

print(rf_IIP.best_estimator_.feature_importances_)
df['iip_est'] = rf_IIP.best_estimator_.predict(X)
heatmapIIP_data = pd.pivot_table(df, values='isInProfit', 

                     index=['pointsAway'], 

                     columns='trailingStop')

heatmapPRED_data = pd.pivot_table(df, values='iip_est', 

                     index=['pointsAway'], 

                     columns='trailingStop')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,6))

sns.heatmap(heatmapIIP_data, ax=ax1, cmap="seismic")

sns.heatmap(heatmapPRED_data, ax=ax2, cmap="seismic")
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics.regression import mean_squared_error



y_FD = df['finishDep']

y_PNC = df['profitNbWithComm']
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y_FD)

rf_FD = GridSearchCV(RandomForestRegressor(n_jobs=-1, random_state=17), 

                                 parameters, 

                                 scoring='neg_mean_squared_error',  

                                 n_jobs=-1, cv=5,

                                  verbose=True)

rf_FD.fit(X_train, y_train)
rf_FD.best_params_, rf_FD.best_score_
print("Mean squared error (cv): %.3f" % np.mean(np.abs(cross_val_score(rf_FD.best_estimator_,

                                                        X_train, y_train, 

                                                        scoring='neg_mean_squared_error'))))

print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, 

                                                             rf_FD.predict(X_holdout)))
rf_FD.best_estimator_.feature_importances_
df['fd_est'] = rf_FD.best_estimator_.predict(X)
heatmapFD_data = pd.pivot_table(df, values='finishDep', 

                     index=['pointsAway'], 

                     columns='trailingStop')

heatmapPRED_data = pd.pivot_table(df, values='fd_est', 

                     index=['pointsAway'], 

                     columns='trailingStop')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,6))

sns.heatmap(heatmapFD_data, ax=ax1, cmap="seismic", center=1000)

sns.heatmap(heatmapPRED_data, ax=ax2, cmap="seismic", center=1000)
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y_PNC)

rf_PNC = GridSearchCV(RandomForestRegressor(n_jobs=-1, random_state=17), 

                                 parameters, 

                                 scoring='neg_mean_squared_error',  

                                 n_jobs=-1, cv=5,

                                  verbose=True)

rf_PNC.fit(X_train, y_train)
rf_PNC.best_params_, rf_PNC.best_score_
rf_PNC.best_estimator_.feature_importances_
df['pnc_est'] = rf_PNC.best_estimator_.predict(X)
heatmapPNC_data = pd.pivot_table(df, values='profitNbWithComm', 

                     index=['pointsAway'], 

                     columns='trailingStop')

heatmapPRED_data = pd.pivot_table(df, values='pnc_est', 

                     index=['pointsAway'], 

                     columns='trailingStop')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,6))

sns.heatmap(heatmapPNC_data, ax=ax1, cmap="seismic", center=1.5)

sns.heatmap(heatmapPRED_data, ax=ax2, cmap="seismic", center=1.5)
# check the max and min of the parameters we want vary

X.describe()
# use logspace for better resolution on smaller scales

# don't extrapolate with estimators, so grid is between the minmax of the test parameters

# logspace for better resolution with smaller numbers

# pointsAway is the most important, it determines where I open my pending orders, so it has the best resolution

x0 = np.logspace(np.log10(1.5), np.log10(30), num=50)

# stopLoss

x1 = np.logspace(np.log10(2), np.log10(12.5), num=6)

# trailingStop

x2 = np.logspace(np.log10(1.5), np.log10(20), num=25)

# breakevenTrigger 0.0 means no break even mechanism, and the breakEvenTrigger distances are between 2 and 12,5

x3 = np.concatenate(([0.0], np.logspace(np.log10(2), np.log10(12.5), num=5)), axis=None)

# breakevenDistance 0.0 means set stopLoss to the open price, so normal logspace

x4 = np.concatenate(([0.0], np.logspace(np.log10(0.5), np.log10(9.6), num=5)), axis=None)

# maxSlippage only want to select combinations where maxSlippage is 50 - I need this feature for the grid, because the eastimators were trained with it.

x5 = [50.0]
# build grid

x0v, x1v, x2v, x3v, x4v, x5v = np.meshgrid(x0, x1, x2, x3, x4, x5, sparse=False)
X_grid = np.array([x0v, x1v, x2v, x3v, x4v, x5v]).reshape(6, -1).T

X_grid.shape
df_grid = pd.DataFrame(X_grid, columns=['pointsAway', 'stopLoss', 'trailingStop', 'breakevenTrigger', 'breakevenDistance', 'maxSlippage'])
df_grid['iip_est'] = rf_IIP.best_estimator_.predict(X_grid)

df_grid['fd_est'] = rf_FD.best_estimator_.predict(X_grid)

df_grid['pnc_est'] = rf_PNC.best_estimator_.predict(X_grid)

df_grid.head()
# only profitable trades

df_grid = df_grid.loc[(df_grid['iip_est'] == True)]

# almost same as above, but for the regression values

df_grid = df_grid.loc[(df_grid['fd_est'] > 1000)]

# breakevenDistance and breakevenTrigger are relative to the open price, a large breakevenDistance would kick out the position

df_grid = df_grid.loc[(df_grid['breakevenDistance'] < df_grid['breakevenTrigger'])]

df_grid.shape
# new metric of profit relative to the open deposite

df_grid['gainRate'] = (df_grid['fd_est']/1000)-1
df_grid['score'] = df_grid['gainRate'] * np.sqrt(df_grid['pnc_est'])
df_grid['prob'] = df_grid['gainRate'] / df_grid['gainRate'].sum()

df_grid.head()
df_sample = df_grid.sample(n=100, weights='prob')
ax = sns.scatterplot(df_sample['trailingStop'], df_sample['pointsAway'])

ax.invert_yaxis()
df_sample.to_csv("trading_sample.csv", index=True)