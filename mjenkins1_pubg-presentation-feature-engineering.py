import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os

import warnings



%matplotlib inline

warnings.filterwarnings("ignore")
# Set the size of the plots 

plt.rcParams["figure.figsize"] = (18,8)

sns.set(rc={'figure.figsize':(18,8)})
data = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")

print("Finished loading data")
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
data = reduce_mem_usage(data)
data.drop(columns=['rankPoints'], inplace=True)
data.dropna(inplace=True)

data.isnull().values.any()
types = ['solo', 'solo-fpp', 'duo', 'duo-fpp', 'squad', 'squad-fpp']

data = data.loc[data['matchType'].isin(types)]
data['matchType'].unique()
data.columns
data['teamSize'] = data.groupby('groupId')['groupId'].transform('count')

data['maxTeamSize'] = data.groupby('matchId')['teamSize'].transform('max')

data['matchSize'] = data.groupby('matchId')['Id'].transform('nunique')
data['killsPerMeter'] = data['kills']/data['walkDistance']

data['killsPerMeter'].fillna(0, inplace=True)

data['killsPerMeter'].replace(np.inf, 0, inplace=True)
data['healsPerMeter'] = data['heals'] / data['walkDistance']

data['healsPerMeter'].fillna(0, inplace=True)

data['healsPerMeter'].replace(np.inf, 0, inplace=True)
data['totalHeals'] = data['heals'] + data['boosts']
data['totalHealsPerMeter'] = data['totalHeals'] / data['walkDistance']

data['totalHealsPerMeter'].fillna(0, inplace=True)

data['totalHealsPerMeter'].replace(np.inf, 0, inplace=True)
data['killsPerSecond'] = data['kills'] / data['matchDuration']

data['killsPerSecond'].fillna(0, inplace=True)

data['killsPerSecond'].replace(np.inf, 0, inplace=True)
data['TotalHealsPerTotalDistance'] = (data['boosts'] + data['heals']) / (data['walkDistance'] + data['rideDistance'] + data['swimDistance'])

data['TotalHealsPerTotalDistance'].fillna(0, inplace=True)

data['TotalHealsPerTotalDistance'].replace(np.inf, 0, inplace=True)
data['killPlacePerMaxPlace'] = data['killPlace'] / data['maxPlace']

data['killPlacePerMaxPlace'].fillna(0, inplace=True)

data['killPlacePerMaxPlace'].replace(np.inf, 0, inplace=True)
data['headShotPerc'] = data['headshotKills'] / data['kills']

data['headShotPerc'].fillna(0, inplace=True)

data['headShotPerc'].replace(np.inf, 0, inplace=True)
data['damagePerSec'] = data['damageDealt'] / data['matchDuration']

data['damagePerSec'].fillna(0, inplace=True)

data['damagePerSec'].replace(np.inf, 0, inplace=True)
data['totalDistance'] = data['rideDistance'] + data['swimDistance'] + data['walkDistance']
data['totalDistancePerSec'] = data['totalDistance'] / data['matchDuration']

data['totalDistancePerSec'].fillna(0, inplace=True)

data['totalDistancePerSec'].replace(np.inf, 0, inplace=True)
data['killStreakKillRatio'] = data['killStreaks'] / data['kills']

data['killStreakKillRatio'].fillna(0, inplace=True)

data['killStreakKillRatio'].replace(np.inf, 0, inplace=True)
data['killPlacePerMaxPlaceRatio'] = data['killPlace'] / data['maxPlace']

data['killPlacePerMaxPlaceRatio'].fillna(0, inplace=True)

data['killPlacePerMaxPlaceRatio'].replace(np.inf, 0, inplace=True)
data['killPlacePerkillPoints'] = data['killPlace'] / data['killPoints']

data['killPlacePerkillPoints'].fillna(0, inplace=True)

data['killPlacePerkillPoints'].replace(np.inf, 0, inplace=True)
data['weaponsAcquiredPerSec'] = data['weaponsAcquired'] / data['matchDuration']

data['weaponsAcquiredPerSec'].fillna(0, inplace=True)

data['weaponsAcquiredPerSec'].replace(np.inf, 0, inplace=True)
data['totalDistancePerWeaponsAcquired'] = data['totalDistance'] / data['weaponsAcquired']

data['totalDistancePerWeaponsAcquired'].fillna(0, inplace=True)

data['totalDistancePerWeaponsAcquired'].replace(np.inf, 0, inplace=True)
data['walkDistancePerHeal'] = data['walkDistance'] / data['heals']

data['walkDistancePerHeal'].fillna(0, inplace=True)

data['walkDistancePerHeal'].replace(np.inf, 0, inplace=True)
data['walkDistancePerKills'] = data['walkDistance'] / data['kills']

data['walkDistancePerKills'].fillna(0, inplace=True)

data['walkDistancePerKills'].replace(np.inf, 0, inplace=True)
data['walkDistancePerSec'] = data['walkDistance'] / data['matchDuration']

data['walkDistancePerSec'].fillna(0, inplace=True)

data['walkDistancePerSec'].replace(np.inf, 0, inplace=True)
data['killEfficiency'] = data['DBNOs'] / data['kills']

data['killEfficiency'].fillna(0, inplace=True)

data['killEfficiency'].replace(np.inf, 0, inplace=True)
data['killPlacePerKills'] = data['killPlace'] / data['kills']

data['killPlacePerKills'].fillna(0, inplace=True)

data['killPlacePerKills'].replace(np.inf, 0, inplace=True)
data['walkDistancePerc'] = data.groupby('matchId')['walkDistance'].rank(pct=True).values
data['total_items_acquired'] = data['boosts'] + data['heals'] + data['weaponsAcquired']
data['playersJoined'] = data.groupby('matchId')['matchId'].transform('count')
data['playersJoined'].unique()
data['killsNormaized'] = data['kills'] / data['playersJoined']
data['killPlacePerc'] = data['killPlace']/ data['maxPlace']

data['killPlacePerc'].fillna(0, inplace=True)

data['killPlacePerc'].replace(np.inf, 0, inplace=True)
data['teamwork'] = data['assists'] + data['revives']
data['damageNormaized'] = data['damageDealt'] / data['playersJoined']
data['killsNoMoving'] = ((data['kills'] > 0) & (data['totalDistance'] == 0))
data['totalDamageByTeam'] = data.groupby('groupId')['damageDealt'].transform('sum')
data['totalKillsByTeam'] =  data.groupby('groupId')['kills'].transform('sum')
len(data.columns)
data.columns
data.drop(['Id', 'groupId', 'matchId',], inplace=True, axis=1)
data.columns
data.info()
data = pd.get_dummies(data, columns=['matchType'], dummy_na=True)
noMoving = [True]

data['killsNoMoving'] = np.where(data['killsNoMoving'].isin(noMoving), 1, 0)
data.info()
from sklearn.metrics import mean_absolute_error as mae

from sklearn.model_selection import train_test_split
data.shape
train, test = train_test_split(data, test_size=0.15, random_state=12)
train.shape
test.shape
train.to_csv('train.csv')
test.to_csv('test.csv')
y = train['winPlacePerc']
X = train

X.drop('winPlacePerc', inplace=True, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=12)
from lightgbm import LGBMRegressor

import datetime
time_0 = datetime.datetime.now()



lgbm = LGBMRegressor(objective='mae', n_jobs=-1, random_state=12)



lgbm.fit(X_train, y_train)



time_1  = datetime.datetime.now()



print('Training took {} seconds.'.format((time_1 - time_0).seconds))

print('Mean Absolute Error is {:.5f}'.format(mae(y_test, lgbm.predict(X_test))))
import shap
SAMPLE_SIZE = 10000

SAMPLE_INDEX = np.random.randint(0, X_test.shape[0], SAMPLE_SIZE)



X = X_test.iloc[SAMPLE_INDEX]



explainer = shap.TreeExplainer(lgbm)

shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type='bar', color='darkred')
# Let's also try xgboost 

import xgboost as xgb
regressor = xgb.XGBRegressor(objective = 'reg:squarederror')

regressor
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# check the MAE

Mae = mae(y_test, y_pred)

print('MAE %f' % (Mae))
xgb.plot_importance(regressor)

plt.title("xgboost.plot_importance(regressor)")

plt.show()