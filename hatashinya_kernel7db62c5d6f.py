# 必要なライブラリのインポート

import numpy as np

import pandas as pd

import math



import pandas_profiling



from sklearn import preprocessing

from category_encoders import OrdinalEncoder



from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss



from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor
# データ読み込み

train = pd.read_csv('../input/exam-for-students20200527/train.csv', encoding='utf-8')

test = pd.read_csv('../input/exam-for-students20200527/test.csv', encoding='utf-8')

gender_submission = pd.read_csv('../input/exam-for-students20200527/sample_submission.csv')

station_info = pd.read_csv('../input/exam-for-students20200527/station_info.csv', encoding='utf-8')

city_info = pd.read_csv('../input/exam-for-students20200527/city_info.csv', encoding='utf-8')



data = pd.concat([train, test], sort=False)
station_info.columns = ['Station', 'st_Latitude', 'st_Longitude']

city_info.columns = ['Prefecture', 'Municipality', 'city_Latitude', 'city_Longitude']
# dataとstation_info, dataとcity_infoをそれぞれ結合する

data = pd.merge(data, station_info, left_on = 'NearestStation', right_on = 'Station', how='left')

data = pd.merge(data, city_info, left_on = ['Prefecture', 'Municipality'], right_on = ['Prefecture', 'Municipality'], how='left')
data.head()
station_info.columns
city_info.columns
train.shape
test.shape
# 学習データの欠損状況

train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
# テストデータの欠損状況

test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
#Type-Region

data['Region'] = data['Region'].fillna('MISSING')

data['Type-Region'] = data['Type'] + '-' + data['Region']



# Year-Quarter

data['Year-Quarter'] = data['Year'].astype(str) + '-' + data['Quarter'].astype(str)



# Remarksフラグ

data['Remarks_flg'] = 1

data.loc[data['Remarks'].isnull(), 'Remarks_flg'] = 0



# MaxTimeToNearestStationが欠損でMinTimeToNearestStationに値がある場合

# data.loc[(data['MaxTimeToNearestStation'].isnull() * data['MinTimeToNearestStation'].notnull()),'MaxTimeToNearestStation'] = -1

data['maxtime-mintime'] = abs(data['MaxTimeToNearestStation'] - data['MinTimeToNearestStation'])





# 駅との距離を計算する

lat_diff = data['st_Latitude'] - data['city_Latitude']

lon_diff = data['st_Longitude'] - data['city_Longitude']

data['kyori'] = np.sqrt(lat_diff ** 2 + lon_diff ** 2)



# 

lat_diff_ike = data['st_Latitude'] - 35.7295384

lon_diff_ike = data['st_Longitude'] - 139.7131303

data['kyori_ike'] = np.sqrt(lat_diff_ike ** 2 + lon_diff_ike ** 2)
# 使わない列を消す

data.drop(['id', 'Remarks', 'TimeToNearestStation'], axis=1, inplace=True)
# 特徴量間の相関を調べる

corr = data.corr()

cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

corr.style.background_gradient(cmap='coolwarm')
data.drop('MaxTimeToNearestStation', axis=1, inplace=True)

# data.drop(['st_Latitude', 'st_Longitude', 'city_Latitude', 'city_Longitude'])
# オブジェクト型の変数を格納

obj_cols = data.dtypes[data.dtypes=='object'].index.tolist()



oe = OrdinalEncoder(cols=obj_cols)

data = oe.fit_transform(data)
obj_cols
train = data[:len(train)]

test = data[len(train):]



y_train = np.log1p(train['TradePrice']) # 評価指標がRMSLEなのでlogとっておく

X_train = train.drop('TradePrice', axis=1)

X_test = test.drop('TradePrice', axis=1)
print(train.shape, test.shape)
# Prefectureをグループ識別子として分離。

groups = X_train['Prefecture'].values



X_train.drop('Prefecture', axis=1, inplace=True)

X_test.drop('Prefecture', axis=1, inplace=True)



groups
groups.max()
n_sp = 5

gkf = GroupKFold(n_splits=n_sp)





seeds = [51,61,71,81,91]

# seeds = [51,61]

scores = []

y_pred_cvavg = np.zeros(len(X_test))



scores = []



for seed in seeds:

    for i, (train_ix, test_ix) in enumerate(gkf.split(X_train, y_train, groups)):

        # 学習データ

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        # 検定データ

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

        clf = LGBMRegressor(n_estimators=9999, random_state=seed, colsample_bytree=0.9,

                            learning_rate=0.05, subsample=0.9, num_leaves=31) 



        

        clf.fit(X_train_, y_train_, early_stopping_rounds=50, eval_metric='rmse', eval_set=[(X_val, y_val)])  



        y_pred = clf.predict(X_val)

        score = np.sqrt(mean_squared_error(y_val, y_pred)) #元データをlog取っているのでRMSLE 

        scores.append(score)



        y_pred_cvavg += clf.predict(X_test)

        print(clf.predict(X_test))

            

        

print(np.mean(scores))

print(scores)



y_pred_cvavg /= n_sp * len(seeds) # fold数 * seed数



y_sub = np.expm1(y_pred_cvavg) # RMSLEなのでexpとる
# 予測結果サンプル

y_sub[:10]
sub = pd.read_csv('../input/exam-for-students20200527/sample_submission.csv')



sub['TradePrice'] = y_sub

sub.to_csv('submission.csv', index=False)



sub.head()
# 特徴量の重要度

imp = pd.DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(by=['importance'], ascending=False)

imp.head(50)
fig, ax = plt.subplots(figsize=(20, 10))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')