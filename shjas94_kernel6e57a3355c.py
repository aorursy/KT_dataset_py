import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from lightgbm import LGBMRegressor

import lightgbm as lgb

from xgboost import XGBRFRegressor

from catboost import CatBoostRegressor, Pool

import shap

import skimage

import os

import gc

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

target = train['winPlacePerc']

train.drop(['winPlacePerc'], axis=1, inplace=True)

merge = pd.concat([train, test], axis=0).reset_index()

train_index = list(range(len(train)))

test_index = list(range(len(train), len(merge)))
merge[merge["kills"] > 35]["kills"] = 35

merge[merge["DBNOs"] > 20]["DBNOs"] = 20

merge[merge["assists"] > 10]["assists"] = 10

merge[merge["walkDistance"] > 10000]["walkDistance"] = 10000

merge[merge["boosts"] > 15]["boosts"] = 15

merge[merge["heals"] > 30]["heals"] = 30

merge[merge["weaponsAcquired"] > 50]["weaponsAcquired"] = 50

merge[merge["damageDealt"] > 2000]["damageDealt"] = 2000

merge[merge["longestKill"] > 500]["longestKill"] = 500

merge[merge["killStreaks"] > 10]["killStreaks"] = 10

merge[merge["rideDistance"] > 15000]["rideDistance"] = 15000

merge[merge["headshotKills"] > 15]["headshotKills"] = 15
def rstr(df, pred=None):

    obs = df.shape[0]

    types = df.dtypes

    counts = df.apply(lambda x: x.count())

    uniques = df.apply(lambda x: [x.unique()])

    nulls = df.apply(lambda x: x.isnull().sum())

    distincts = df.apply(lambda x: x.unique().shape[0])

    missing_ration = (df.isnull().sum()/obs) *100

    skewness = df.skew()

    kurtosis = df.kurt()

    print('Data shape: ', df.shape)

    

    if pred is None:

        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis', 'corr']

        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis=1)

    else:

        corr = df.corr()[pred]

        str =pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis=1, sort=False)

        corr_col = 'corr ' + pred

        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis',  corr_col]

    str.columns = cols

    dtypes = str.types.value_counts()

    print('___________________________\nData types:\n',str.types.value_counts())

    print('___________________________')

    return str
# merge["matchType"].replace(['crashfpp','crashtpp','flaretpp','flarefpp','duo-fpp','normal-duo','normal-duo-fpp','solo-fpp','normal-solo','normal-solo-fpp','squad-fpp','normal-squad','normal-squad-fpp'],

#                            ['event','event','event','event','duo','duo','duo','solo','solo','solo','squad','squad','squad'], inplace=True)
# merge["matchType"].replace(['crashfpp','crashtpp','flaretpp','flarefpp','duo-fpp','normal-duo','normal-duo-fpp','solo-fpp','normal-solo','normal-solo-fpp','squad-fpp','normal-squad','normal-squad-fpp'],

#                            [1,2,3,4,5,6,7,8,9,10,11,12,13], inplace=True)
# merge["matchType"].replace(['duo-fpp','normal-duo','normal-duo-fpp','solo-fpp','normal-solo','normal-solo-fpp','squad-fpp','normal-squad','normal-squad-fpp'],

#                            ['duofpp','normalduo','normalduofpp','solofpp','normalsolo','normalsolofpp','squadfpp','normalsquad','normalsquadfpp'], inplace=True)
matchType = pd.get_dummies(merge.matchType)
merge = pd.concat([merge,matchType], axis=1)
merge["group_size"] = merge.groupby(["matchId"])['groupId'].transform('count')

merge["group_per"] = merge["maxPlace"]/merge["group_size"]
merge["distanceSum"] = 0.85*merge['walkDistance'] + 0.14*merge['rideDistance'] + 0.01*merge['swimDistance']

# merge["distanceMean"] = merge.groupby(["groupId"])['distanceSum'].transform(np.mean)

merge["distanceRank"] = merge.groupby(["matchId"])["distanceSum"].rank(pct=True)

merge.drop("distanceSum", axis=1, inplace=True)
merge["killPlace"] = merge["killPlace"]-merge["group_per"]

merge["killPlace"] = -1*merge["killPlace"]/merge["group_size"]
# merge["killsAndassists"] = merge["kills"]+merge["assists"]

merge["DBNOsNassists"] = merge["DBNOs"]+merge["assists"]

merge["items"] = merge["boosts"]+merge["heals"]

merge["itemsRank"] = merge.groupby(["matchId"])["items"].rank(pct=True)

merge["damageDealtRank"] = merge.groupby(["matchId"])["damageDealt"].rank(pct=True)

merge["damageDealtPerkills"] = merge["damageDealt"]/(merge["kills"]+1)

merge["killStreaksPerkills"] = merge["killStreaks"] / (merge["kills"]+1)

merge["headshotPerKills"] = merge["headshotKills"]/(merge["kills"]+1)

merge["weaponsAcqRank"] = merge.groupby(["matchId"])["weaponsAcquired"].rank(pct=True)

merge["longestKillRank"] = merge.groupby(["matchId"])["longestKill"].rank(pct=True)

merge["DBNOsNassistsRank"] = merge.groupby(["matchId"])["DBNOsNassists"].rank(pct=True)

merge["revivesRank"] = merge.groupby(["matchId"])["revives"].rank(pct=True)

merge.drop(["boosts", "heals", "assists", "damageDealt", "killStreaks", "headshotKills", "weaponsAcquired", "longestKill", "DBNOs", "revives"],

          axis=1, inplace=True)
merge["itemsRank"] = merge.groupby(["groupId"])["items"].transform(np.mean)

merge["damageDealtRank"] = merge.groupby(["groupId"])["damageDealtRank"].transform(np.mean)

merge["damageDealtPerkills"] = merge.groupby(["groupId"])["damageDealtPerkills"].transform(np.mean)

merge["killStreaksPerkills"] = merge.groupby(["groupId"])["killStreaksPerkills"].transform(np.mean)

merge["headshotPerKills"] = merge.groupby(["groupId"])["headshotPerKills"].transform(np.mean)

merge["weaponsAcqRank"] = merge.groupby(["groupId"])["weaponsAcqRank"].transform(np.mean)

merge["longestKillRank"] = merge.groupby(["groupId"])["longestKillRank"].transform(np.mean)

merge["DBNOsNassistsRank"] = merge.groupby(["groupId"])["DBNOsNassistsRank"].transform(np.mean)

merge["revivesRank"] = merge.groupby(["groupId"])["revivesRank"].transform(np.mean)

merge["distanceRank"] = merge.groupby(["groupId"])["distanceRank"].transform(np.mean)

merge["killStreaksPerkillsRank"] = merge.groupby(["matchId"])["killStreaksPerkills"].rank(pct=True)

merge["headshotPerKillsRank"] = merge.groupby(["matchiId"])["headshotPerKills"].rank(pct=True)
# merge["walkDistanceMean"] = merge.groupby(["groupId"])["walkDistance"].transform(np.mean)

# merge["killMean"] = merge.groupby(["groupId"])["kills"].transform(np.mean)

# merge["group_size"] = merge.groupby(["matchId"])['groupId'].transform('count')

# merge["group_per"] = merge["maxPlace"]/merge["group_size"]



# merge["killPlace"] = merge["killPlace"]-merge["group_per"]

# merge['killPlaceMean'] = merge.groupby(["groupId"])["killPlace"].transform(np.mean)

# merge["killPlaceMean"] = -1*merge["killPlaceMean"]/merge["group_size"]



# merge["distanceSum"] = 0.85*merge['walkDistance'] + 0.14*merge['rideDistance'] + 0.01*merge['swimDistance']

# merge["distanceMean"] = merge.groupby(["groupId"])['distanceSum'].transform(np.mean)

# merge["distanceMean"] = np.cbrt(merge["distanceMean"]/(merge["matchDuration"]))



# merge["damageDealtMean"] = merge.groupby(["groupId"])["damageDealt"].transform(np.mean)

# merge['damageDealtPerwalkDistance'] = np.log10(merge['damageDealtMean']/(merge["walkDistanceMean"]))

# merge["damageDealtPerwalkDistance"].replace([-np.inf, np.inf], np.nan, inplace=True)

# merge["damageDealtPerwalkDistance"].fillna(0, inplace=True)





# merge["items"] = 0.6*merge["boosts"]+0.4*merge["heals"]

# merge["items"] = merge.groupby(["groupId"])["items"].transform(np.mean)



# merge["itemsPerwalkDistance"] = np.log10(merge["items"]/(merge["walkDistanceMean"]))

# merge["itemsPerwalkDistance"].replace([-np.inf, np.inf], np.nan, inplace=True)

# merge["itemsPerwalkDistance"].fillna(0, inplace=True)



# merge["itemsPerDamage"] = np.log10(merge["items"]/(merge["damageDealtMean"]+1))

# merge["itemsPerDamage"].replace([-np.inf, np.inf], np.nan, inplace=True)

# merge["itemsPerDamage"].fillna(0, inplace=True)



# merge["itemsPerKills"] = merge["items"]/(merge["kills"]+1)





# merge["Rampage"] = 0.58*merge["killStreaks"]+0.42*merge["headshotKills"]

# merge["Rampage"] = merge.groupby(["groupId"])["Rampage"].transform(np.mean)

# merge["RampagePerwalkDistance"] = np.log10(merge["Rampage"]/(merge["walkDistanceMean"]))

# merge['RampagePerwalkDistance'].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge['RampagePerwalkDistance'].fillna(0, inplace=True)

# merge["RampagePerKills"] = merge["Rampage"]/(merge["killMean"]+1)

# merge["killsPerRampage"] = merge["killMean"]/(merge["Rampage"]+1)

# merge["RampagePerDuration"] = merge["Rampage"]/merge["matchDuration"]



# merge["kills"] = 0.99*merge["kills"] + 0.01*merge["roadKills"]

# merge["kills"] = merge.groupby(["groupId"])["kills"].transform(np.mean) #

# merge["killsPermatchDuration"] = merge["kills"]/(merge["matchDuration"]+1)

# merge['killsPermatchDuration'].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge['killsPermatchDuration'].fillna(0, inplace=True)

# merge["killsPerwalkDistance"] = np.log10(merge["kills"]/(merge["walkDistanceMean"]))

# merge['killsPerwalkDistance'].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge['killsPerwalkDistance'].fillna(0, inplace=True)

# merge["walkDistancePerkills"] = merge["walkDistanceMean"]/(merge["killMean"]+1)



# merge["DBNOsNassists"] = 0.4*merge["DBNOs"]+0.6*merge["assists"]

# merge["DBNOsNassists"] = merge.groupby(["groupId"])["DBNOsNassists"].transform(np.mean)

# merge["DBNOsNassistsPerwalk"] = np.log10(merge["DBNOsNassists"]/(merge["walkDistanceMean"]))

# merge["DBNOsNassistsPerwalk"].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge["DBNOsNassistsPerwalk"].fillna(0, inplace=True)

# merge["walkPerDBNOsNassists"] = merge["walkDistance"]/(merge["DBNOsNassists"]+1)

# merge["DBNOsNassistsPerkills"] = np.cbrt(merge["DBNOsNassists"]/(merge["killMean"]+1))

# merge["DBNOsNassistsPerkills"].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge["DBNOsNassistsPerkills"].fillna(0, inplace=True)

# merge["killsPerDBNOsNassists"] = np.cbrt(merge["killMean"]/(merge["DBNOsNassists"]+1))

# merge["killsPerDBNOsNassists"].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge["killsPerDBNOsNassists"].fillna(0, inplace=True)

# merge["DBNOsNassistsPerDuration"] = merge["DBNOsNassists"]/merge["matchDuration"]



# merge["revives"] = merge.groupby(["groupId"])["revives"].transform(np.mean) #

# merge['revives'].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge['revives'].fillna(0, inplace=True)

# merge['revivesPerwalk'] = np.log10(merge["revives"]/(merge["walkDistanceMean"]))

# merge['revivesPerwalk'].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge['revivesPerwalk'].fillna(0, inplace=True)

# merge["revivesPermatchDuration"] = np.cbrt(merge["revives"]/(merge["matchDuration"]+1))





# merge["meanWeapon"] = merge.groupby(["groupId"])["weaponsAcquired"].transform(np.mean)

# merge["meanWeaponPerwalkDistance"] = np.log10(merge["meanWeapon"] / (merge["walkDistanceMean"]))

# merge['meanWeaponPerwalkDistance'].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge['meanWeaponPerwalkDistance'].fillna(0, inplace=True)

# merge["meanWeaponPerDuration"] = merge["meanWeapon"]/merge["matchDuration"]



# merge['longestKill'] = merge.groupby(["groupId"])["longestKill"].transform(np.mean)

# merge['longestKill'] = np.sqrt(merge["longestKill"])

# merge['longestKill'].replace([np.inf, -np.inf], np.nan, inplace=True)

# merge['longestKill'].fillna(0, inplace=True)

# etc = merge[['numGroups', 'Id', 'groupId','matchId', 'matchType']]





# merge.drop(['vehicleDestroys','rankPoints','killPoints','winPoints','matchDuration', 'killMean', 'Rampage','group_per',

#            'roadKills','teamKills','maxPlace','DBNOs', 'assists','walkDistanceMean','boosts','heals','killStreaks','headshotKills',

#            'group_size','walkDistance', 'rideDistance','swimDistance','damageDealt','killPlace','distanceSum','weaponsAcquired'], axis=1, inplace=True)
merge_id = merge['Id']

merge.drop(['Id','groupId','matchId','numGroups','matchType'], axis=1, inplace=True)
merge.columns
train = merge.loc[train_index,:]

test = merge.loc[test_index,:].reset_index()
train.drop(['index'], axis=1, inplace=True)

test.drop(['index', 'level_0'], axis=1, inplace=True)
train = pd.concat([train, target],axis=1)
# pd.set_option('display.max_rows', None)

# details = rstr(train, 'winPlacePerc')

# display(details.sort_values(by='corr winPlacePerc', ascending=False))
target = train['winPlacePerc']

target = target.fillna(target.mean())

train.drop('winPlacePerc', axis=1, inplace=True)
del merge

gc.collect()
x_train, x_test, y_train, y_test = train_test_split(train, target, train_size = 0.9, test_size = 0.1, random_state = 0) ## train,test size arranged
test_ds = Pool(data=x_test, label=y_test)

train_ds = Pool(data=x_train, label=y_train)

# lgtrain = lgb.Dataset(x_train, label=y_train)

# lgval = lgb.Dataset(x_test, label=y_test)
# del x_test, y_test

# gc.collect()
# params = {'objective':'regression_l1', 

#           'boosting':'gbdt', 'metric':'mae',

#           'max_depth':14, 'learning_rate':0.01, 'bagging_fraction':0.8,

#           'feature_fraction':0.8, 'device_type':'gpu', 'n_estimators' : 8000, 'verbosity':-1,'early_stopping_round':200,'verbose_eval':100}
lr = LinearRegression()

ridge = Ridge(alpha = 1.0)

# Xgb = XGBRegressor(max_depth=12,n_jobs=-1,n_estimators=2000,tree_method='gpu_hist',eval_metric='mae',

#                    sampling_method='gradient_based',reg_lambda=2.0,learning_rate=0.5, subsample=0.8)

# Xgb = XGBRegressor(max_depth=14,n_jobs=-1,n_estimators=2000,tree_learner='gpu_hist',eval_metric='mae', booster='gbtree',gpu_id=0,

#                    sampling_method='gradient_based',reg_lambda=1.5,subsample=0.8, learning_rate=0.01, objective='reg:squarederror')

# lgbm = LGBMRegressor(objective='regression_l1', boosting='gbdt', metric='mae',

#                      max_depth=14, learning_rate=0.01, bagging_fraction=0.8,

#                     feature_fraction=0.8, device_type='gpu', n_estimators = 8000, verbosity=-1)

 

# model = LGBMRegressor.fit(params, x_train, y_train ,eval_set=test_ds, early_stopping_rounds=7000, eval_metric='mae')

# Cat = CatBoostRegressor(max_leaves=140, n_estimators=2000, loss_function='MAE', eval_metric='MAE',learning_rate=0.01,

#                         task_type='GPU',l2_leaf_reg=3.0, use_best_model=True)

lgbreg = LGBMRegressor(objective='regression_l1', 

          boosting='gbdt', metric='mae',

          max_leaves=250, learning_rate=0.01, bagging_fraction=0.7,

          feature_fraction=0.7, device_type='gpu', n_estimators=5000, verbosity=100,early_stopping_round=100,verbose_eval=100)
# lr_model = lr.fit(x_train, y_train)

# ridge_model = ridge.fit(x_train, y_train)

# xgb_model = Xgb.fit(x_train, y_train)

lgb_model = lgbreg.fit(x_train, y_train, eval_metric='mae',eval_set=[(x_test, y_test)])

# cat_model = Cat.fit(train_ds, eval_set=test_ds, early_stopping_rounds=200, metric_period=100)
# print("훈련 스코어(lr)     : %.4f" % lr_model.score(x_train, y_train))

# print("훈련 스코어(ridge)  : %.4f" % ridge_model.score(x_train, y_train))

# print("훈련 스코어(xgb)    : %.4f" % xgb_model.score(x_train, y_train))

# print("훈련 스코어(lgb)    : %.4f" % lgb_model.score(x_train, y_train))
# print("훈련 mae(lr)       : %.4f" % mean_absolute_error(y_train, lr_model.predict(x_train)))

# print("훈련 mae(ridge)    : %.4f" % mean_absolute_error(y_train, ridge_model.predict(x_train)))

# print("훈련 mae(xgb)      : %.4f" % mean_absolute_error(y_train, xgb_model.predict(x_train)))

# print("훈련 mae(lgb)      : %.4f" % mean_absolute_error(y_train, lgb_model.predict(x_train)))

# print("훈련 mae(cat)      : %.4f" % mean_absolute_error(y_train, cat_model.predict(x_train)))
# print("예측 스코어(lr)     : %.4f" % r2_score(y_test, lr_model.predict(x_test))) 

# print("예측 스코어(ridge)  : %.4f" % r2_score(y_test, ridge_model.predict(x_test)))

# print("예측 스코어(xgb)    : %.4f" % r2_score(y_test, xgb_model.predict(x_test)))

# print("예측 스코어(lgb)    : %.4f" % r2_score(y_test, predict))
# print("예측 mae(lr)     : %.4f" % mean_absolute_error(y_test, lr_model.predict(x_test))) 

# print("예측 mae(ridge)  : %.4f" % mean_absolute_error(y_test, ridge_model.predict(x_test))) 

# print("예측 mae(xgb)    : %.4f" % mean_absolute_error(y_test, xgb_model.predict(x_test)))

# print("예측 mae(lgb)    : %.4f" % mean_absolute_error(y_test, lgb_model.predict(x_test)))

# print("예측 mae(cat)    : %.4f" % mean_absolute_error(y_test, cat_model.predict(x_test)))
# fea_imp = pd.DataFrame({'imp': cat_model.feature_importances_, 'col': x_train.columns})

# fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]

# fea_imp.plot(kind='barh', x='col', y='imp', figsize=(15, 15), legend=None, )

# plt.title('CatBoost - Feature Importance')

# plt.ylabel('Features')

# plt.xlabel('Importance')
# explainer = shap.TreeExplainer(cat_model)

# shap_values = explainer.shap_values(x_test)
# # Xgb_model_final = Xgb.fit(train, target)

# lgbm_final = LGBMRegressor(learning_rate=0.05, n_estimators=2000, max_depth=13, boosting_type='gbdt', objective='regression_l1',

#                     device_type='gpu', metric='mae', num_iterations=2000, feature_fraction=0.7, bagging_fraction=0.8, max_cat_threshold=30)

# answer = lgbm_final.fit(train, target)





test_v2 =  pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

Id = test_v2['Id']

test_v2.drop('Id', inplace=True, axis=1)

test.head()
del test_v2

gc.collect()
del x_train, y_train

gc.collect()
answer_final = lgb_model.predict(test)

submission = pd.DataFrame({'Id':Id, 'winPlacePerc':answer_final})

submission.tail()
idx_lower_bound = submission[submission['winPlacePerc'] < 0].index

submission.iloc[idx_lower_bound,1] = 0 

idx_upper_bound = submission[submission['winPlacePerc'] > 1].index

submission.iloc[idx_upper_bound,1] = 1 

submission[submission['winPlacePerc'] < 0].count()

submission[submission['winPlacePerc'] >1].count()

submission.to_csv("submission.csv", index=False)