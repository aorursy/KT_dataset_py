import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.style.use('seaborn-notebook')

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools

init_notebook_mode(connected=True)

import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
import cufflinks as cf
cf.go_offline()
print('Max damage:', train['damageDealt'].max())
print('95% percentile:', np.percentile(train['damageDealt'], 95))
print('99% percentile:', np.percentile(train['damageDealt'], 99))
print('{0:.4f}% players dealt zero damage'.format((train['damageDealt'] == 0).sum()/ train.shape[0]))
plt.figure(figsize=(12, 8))
plt.hist(train.loc[train['damageDealt'] <= 800, 'damageDealt'], bins=40);
plt.title('Distribution of damage dealt without outliers.');
train.damageDealt.value_counts().head().iplot(kind='bar', title='Top 5 most common values of dealt damage')
print('Distribution of number of kills')
print('Max number of kills:', train['DBNOs'].max())
print('95% percentile:', np.percentile(train['DBNOs'], 95))
print('99% percentile:', np.percentile(train['DBNOs'], 99))
print('{0:.4f}% players killed noone'.format((train['DBNOs'] == 0).sum()/ train.shape[0]))
plt.hist(train['DBNOs'], bins=40);
train.DBNOs.value_counts().head().iplot(kind='bar', title='Top 5 most common values of number of kills')
plt.scatter(x=train['rideDistance'], y=train['roadKills']);
plt.scatter(x=train.loc[train['rideDistance'] <= 20000, 'rideDistance'], y=train.loc[train['rideDistance'] <= 20000, 'roadKills']);
train['winPlacePerc'].plot(kind='hist');
plt.title('Distibution of target.');
plt.hist(train['killPlace'], bins=20);
plt.title('Distribution of kill place.')
X = train[['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']]
X['walkDistance_to_mean'] = X['walkDistance'] / X['walkDistance'].mean()
X['walkDistance_to_лшдды'] = X['walkDistance'] / X['kills']
y = train['winPlacePerc']
X_test = test[['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']]
X_test['walkDistance_to_mean'] = X_test['walkDistance'] / X_test['walkDistance'].mean()
X_test['walkDistance_to_лшдды'] = X_test['walkDistance'] / X_test['kills']
params = {"objective" : "regression", "metric" : "mae", "max_depth": 5, "min_child_samples": 20, "reg_alpha": 0.2, "reg_lambda": 0.2,
        "num_leaves" : 33, "learning_rate" : 0.2, "subsample" : 0.9, "colsample_bytree" : 0.9, "subsample_freq ": 6}
n_fold = 10
folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)
# Cleaning and defining parameters for LGBM
model = lgb.LGBMRegressor(**params, n_estimators = 5000, nthread = 4, n_jobs = -1)
prediction = np.zeros(test.shape[0])
scores = []
for fold_n, (train_index, test_index) in enumerate(folds.split(X)):
    print('Fold:', fold_n)
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
            verbose=500, early_stopping_rounds=100)
    
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    prediction += y_pred
    
    scores.append(mean_absolute_error(y_valid, model.predict(X_valid)))
prediction /= n_fold
print(f'Mean CV: {np.mean(scores):.4f}. Std: {np.std(scores):.4f}')
lgb.plot_importance(model, max_num_features=30, figsize=(12, 8));
plt.title('Feature importance');
submission = pd.read_csv('../input/sample_submission.csv')
submission['winPlacePerc'] = np.clip(prediction, 0, 1)
submission.to_csv('sub.csv', index=False)