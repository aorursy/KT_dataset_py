# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings('ignore')
train=pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
test=pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
sample_submission=pd.read_csv('../input/pubg-finish-placement-prediction/sample_submission_V2.csv')
train.head()
test.head()
print(train.shape, test.shape)
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
train[train['winPlacePerc'].isnull()]
train.drop(2744604, inplace=True)
train.shape
train['killsPerWalkDistance'] = train['kills']/(train['walkDistance']+1)
train[['kills', 'walkDistance', 'rideDistance', 'killsPerWalkDistance', 'winPlacePerc']].sort_values(by='killsPerWalkDistance').tail(10)
abnormalities_train=train[(train['walkDistance']==0)&(train['rideDistance']==0)][train['kills']>1]
indexes_train=abnormalities_train.index.values
indexes_train
train.drop(index=indexes_train,inplace=True)
print(train.shape)
abnormalities_test=test[(test['walkDistance']==0)&(test['rideDistance']==0)][test['kills']>1]
indexes_test=abnormalities_test.index.values
indexes_test
test.drop(index=indexes_test,inplace=True)
print(test.shape)
train_data=train[(train['winPlacePerc']<1)]
print(train_data.shape)
sns.distplot(train_data['kills'][:], rug=True)
plt.title('Distribution of kills', fontsize=15)
plt.show()
plt.figure(figsize=(15,8))
ax = sns.boxplot(x="kills",y="damageDealt", data = train_data)
ax.set_title("Damage dealt vs. Number of Kills")
plt.show()
winperc=train_data.groupby('matchType').winPlacePerc.mean()
winperc.plot(kind='bar',figsize=(15,8))
plt.title('Performance based on match-type',fontsize=18)
plt.xticks(rotation=60)
plt.show()
fig,ax=plt.subplots(figsize=(15,12))
ax=sns.heatmap(train_data.corr(),annot=True)
corr=train_data.corr()
round(corr,3)
train_data['healsAndBoosts'] = train_data['heals']+train_data['boosts']
train_data['totalDistance'] = train_data['walkDistance']+train_data['rideDistance']+train_data['swimDistance']
train_data['DamageRate'] = train_data['damageDealt']/(train_data['DBNOs']+1)
train_data['avg_ranking'] = (train_data['killPoints']+train_data['rankPoints']+train_data['winPoints'])/3
test['healsAndBoosts'] = test['heals']+test['boosts']
test['totalDistance'] = test['walkDistance']+test['rideDistance']+test['swimDistance']
test['DamageRate'] = test['damageDealt']/(test['DBNOs']+1)
test['avg_ranking'] = (test['killPoints']+test['rankPoints']+test['winPoints'])/3
x_train=train_data[["assists","healsAndBoosts","DamageRate","killPlace","avg_ranking","kills","longestKill","matchDuration","numGroups","revives","totalDistance","teamKills","vehicleDestroys","weaponsAcquired"]]
y_train=train_data['winPlacePerc']
x_test=test[["assists","healsAndBoosts","DamageRate","killPlace","avg_ranking","kills","longestKill","matchDuration","numGroups","revives","totalDistance","teamKills","vehicleDestroys","weaponsAcquired"]]
print(x_train.shape, y_train.shape, x_test.shape)
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler=MinMaxScaler()
x_train_mms=scaler.fit_transform(x_train)
x_test_mms=scaler.transform(x_test)
import xgboost as xgb
model_xgb = xgb.XGBRegressor(max_depth=19, gamma=0.3, learning_rate= 0.1, tree_method='exact', n_estimators=100)
model_xgb.fit(x_train_mms,y_train)
from sklearn.metrics import mean_absolute_error
print('MAE :', mean_absolute_error(y_train,model_xgb.predict(x_train_mms)))
xgb.plot_importance(model_xgb)
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_jobs=-1, n_estimators = 25, max_leaf_nodes=10000, random_state=1)
model_rf.fit(x_train,y_train)
print('MAE :', mean_absolute_error(y_train,model_rf.predict(x_train)))
from sklearn.linear_model import LinearRegression
linear_model=LinearRegression(n_jobs=-1)
linear_model.fit(x_train_mms,y_train)
print('MAE :', mean_absolute_error(y_train,linear_model.predict(x_train_mms)))
import lightgbm as lgb
model_gbm = lgb.LGBMRegressor(bagging_fraction=0.7, bagging_freq=10, boosting_type='gbdt',
       class_weight=None, colsample_bytree=0.5, feature_fraction=0.9,
       importance_type='split', learning_rate=0.03, max_bin=512,
       max_depth=8, metric='mae', min_child_samples=20,
       min_child_weight=0.001, min_split_gain=0.0, n_estimators=1000,
       n_jobs=-1, num_leaves=150, objective='regression', random_state=None, reg_alpha=0.0,
       reg_lambda=0.0, silent=True, task='train', verbose=0)
model_gbm.fit(x_train_mms,y_train)
print('MAE :', mean_absolute_error(y_train,model_gbm.predict(x_train_mms)))
winperc_predict=model_xgb.predict(x_test_mms)
output = pd.DataFrame({'Id': test.Id,
                       'winPlacePerc': winperc_predict})
output.to_csv("./submission.csv", index=False)