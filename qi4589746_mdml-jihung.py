import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



# 用来绘图的，封装了matplot

# 要注意的是一旦导入了seaborn，

# matplotlib的默认作图风格就会被覆盖成seaborn的格式

import seaborn as sns       



from scipy import stats

from scipy.stats import  norm

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline  
# data_train = pd.read_csv('../input/train.csv')
# data_train['parking_area'] = data_train['parking_area'].fillna(0)

# # data_train['parking_area'] = (data_train['parking_area'] > 0).astype(int)



# data_train['parking_price'] = data_train['parking_price'].fillna(0)

# # data_train['parking_price'] = (data_train['parking_price'] > 0).astype(int)



# data_train['txn_floor'] = data_train['txn_floor'].fillna(0)

# # data_train['txn_floor'] = (data_train['txn_floor'] > 0).astype(int)



# data_train['village_income_median'] = data_train['village_income_median'].fillna(0)

# # data_train['village_income_median'] = (data_train['village_income_median'] > 0).astype(int)

# data_train['total_price'].describe()
# corrmat1 = data_train.corr()
# k  = 15 # 关系矩阵中将显示10个特征

# cols = corrmat1.nlargest(k, 'total_price')['total_price'].index

# cm = np.corrcoef(data_train[cols].values.T)

# sns.set(font_scale=1.25)

# hm = sns.heatmap(cm, cbar=True, annot=True, \

#                  square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)

# plt.show()



# ['master_rate','doc_rate','bachelor_rate','jobschool_rate','total_floor','XIII_5000','born_rate','II_1000','lon','VII_1000']
# k  = 15 # 关系矩阵中将显示10个特征

# cols = corrmat.nlargest(k, 'village_income_median')['village_income_median'].index

# cm = np.corrcoef(data_train[cols].values.T)

# sns.set(font_scale=1.25)

# hm = sns.heatmap(cm, cbar=True, annot=True, \

#                  square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)

# plt.show()



# ['master_rate','doc_rate','bachelor_rate','jobschool_rate','total_floor','XIII_5000','born_rate','II_1000','lon','VII_1000']

# sns.set()

# cols = ['village_income_median','master_rate','doc_rate','bachelor_rate','jobschool_rate','total_floor','XIII_5000','born_rate','II_1000','lon','VII_1000']

# sns.pairplot(data_train[cols], size = 2.5)

# plt.show()
# k  = 15 # 关系矩阵中将显示10个特征

# cols = corrmat.nlargest(k, 'txn_floor')['txn_floor'].index

# cm = np.corrcoef(data_train[cols].values.T)

# sns.set(font_scale=1.25)

# hm = sns.heatmap(cm, cbar=True, annot=True, \

#                  square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)

# plt.show()



# ['total_floor','N_50','building_material','building_complete_dt','XIII_index_1000','XIII_index_50','N_500','N_1000','VII_index_50','VIII_250']

# sns.set()

# cols = ['txn_floor','total_floor','N_50','building_material','building_complete_dt','XIII_index_1000','XIII_index_50','N_500','N_1000','VII_index_50','VIII_250']

# sns.pairplot(data_train[cols], size = 2.5)

# plt.show()
# k  = 15 # 关系矩阵中将显示10个特征

# cols = corrmat.nlargest(k, 'parking_price')['parking_price'].index

# cm = np.corrcoef(data_train[cols].values.T)

# sns.set(font_scale=1.25)

# hm = sns.heatmap(cm, cbar=True, annot=True, \

#                  square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)

# plt.show()



# ['total_floor','building_complete_dt','jobschool_rate','lon','XIII_5000','bachelor_rate','XIII_10000','VII_10000','V_10000', 'master_rate']

# sns.set()

# cols = ['total_floor','building_complete_dt','jobschool_rate','lon','XIII_5000','bachelor_rate','XIII_10000','VII_10000','V_10000', 'master_rate']

# sns.pairplot(data_train[cols], size = 2.5)

# plt.show()
# k  = 15 # 关系矩阵中将显示10个特征

# cols = corrmat.nlargest(k, 'parking_area')['parking_area'].index

# cm = np.corrcoef(data_train[cols].values.T)

# sns.set(font_scale=1.25)

# hm = sns.heatmap(cm, cbar=True, annot=True, \

#                  square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)

# plt.show()



# ['building_area','total_floor','building_complete_dt','lat','lon','marriage_rate','born_rate','building_material','jobschool_rate', 'bachelor_rate']
# sns.set()

# cols = ['building_area','total_floor','building_complete_dt','lat','lon','marriage_rate','born_rate','building_material','jobschool_rate', 'bachelor_rate']

# sns.pairplot(data_train[cols], size = 2.5)

# plt.show()
from sklearn import preprocessing

from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import xgboost

import numpy as np

data_train2 = pd.read_csv('../input/train.csv')

# data_train2 = pd.read_csv('../input/train.csv', index_col=0)



import lightgbm as lgb
# 補 parking_area

parking_area_train = data_train2.loc[~pd.isna(data_train2['parking_area'])]

parking_area_train



parking_area_train_cols = ['building_area','total_floor','building_complete_dt','lat','lon','marriage_rate','born_rate','building_material','jobschool_rate', 'bachelor_rate']

# parking_area_train_cols = ['building_area','total_floor','building_complete_dt','lat','lon']



parking_area_train_x = parking_area_train[parking_area_train_cols].values

parking_area_train_y = parking_area_train['parking_area'].values



# --------



parking_area_train_x_scaled = preprocessing.StandardScaler().fit_transform(parking_area_train_x)

parking_area_train_y_scaled = preprocessing.StandardScaler().fit_transform(parking_area_train_y.reshape(-1,1))

parking_area_train_X_train,parking_area_train_X_test, parking_area_train_y_train, parking_area_train_y_test = train_test_split(parking_area_train_x_scaled, parking_area_train_y_scaled, test_size=0.33, random_state=42)



# parking_area_train_clfs = {

#         'svm':svm.SVR(), #4

#         'RandomForestRegressor':RandomForestRegressor(n_estimators=400), #1

#         'BayesianRidge':linear_model.BayesianRidge(), #3

#         'xgboost' :xgboost.XGBRegressor(learning_rate=0.01,n_estimators=3460,

#                                      max_depth=3, min_child_weight=0,

#                                      gamma=0, subsample=0.7,

#                                      colsample_bytree=0.7,

#                                      objective='reg:linear', nthread=-1,

#                                      scale_pos_weight=1, seed=27,

#                                      reg_alpha=0.00006),

#         'lgb':lgb.LGBMRegressor(objective='regression', 

#                                        num_leaves=20, 

#                                        learning_rate=0.05, 

#                                        n_estimators=5000,

#                                        max_bin=200, 

#                                        bagging_fraction=0.75,

#                                        bagging_freq=5, 

#                                        bagging_seed=7,

#                                        feature_fraction=0.2, 

#                                        feature_fraction_seed=7,

#                                        verbose=-1,

#                                        num_thread=6,) #2

#        }

# for clf in parking_area_train_clfs:

#     try:

#         parking_area_train_clfs[clf].fit(parking_area_train_X_train, parking_area_train_y_train)

#         parking_area_train_y_pred = parking_area_train_clfs[clf].predict(parking_area_train_X_test)

#         print(clf + " cost:" + str(np.sum(parking_area_train_y_pred-parking_area_train_y_test)/len(parking_area_train_y_pred)) )

#     except Exception as e:

#         print(clf + " Error:")

#         print(str(e))



# --------
# parking_area_train_X_train,parking_area_train_X_test, parking_area_train_y_train, parking_area_train_y_test = train_test_split(parking_area_train_x, parking_area_train_y, test_size=0.33, random_state=42)

parking_area_train_X_train,parking_area_train_X_test, parking_area_train_y_train, parking_area_train_y_test = train_test_split(parking_area_train_x_scaled, parking_area_train_y, test_size=0.33, random_state=42)

parking_area_train_clt = RandomForestRegressor(n_estimators=400)

parking_area_train_clt.fit(parking_area_train_X_train, parking_area_train_y_train)



parking_area_has_na = data_train2.loc[pd.isna(data_train2['parking_area'])]

parking_area_has_na.drop(['parking_area'], 1, inplace=True)

parking_area_has_na_X = parking_area_has_na[parking_area_train_cols].values

print('parking_area predicting...')

parking_area_has_na_Y = parking_area_train_clt.predict(parking_area_has_na_X)

print('OK')



parking_area_prediction = pd.DataFrame(parking_area_has_na_Y, columns=['parking_area'])

parking_area_has_na = parking_area_has_na.reset_index()

parking_area_result = pd.concat([parking_area_has_na, parking_area_prediction],axis = 1)

parking_area_result = parking_area_result.set_index('building_id')



# parking_area_result
# 補 parking_price

parking_price_train = data_train2.loc[~pd.isna(data_train2['parking_price'])]

parking_price_train



parking_price_train_cols = ['total_floor','building_complete_dt','jobschool_rate','lon','XIII_5000','bachelor_rate','XIII_10000','VII_10000','V_10000', 'master_rate']

# parking_price_train_cols = ['total_floor','building_complete_dt','jobschool_rate','lon','XIII_5000']



parking_price_train_x = parking_price_train[parking_price_train_cols].values

parking_price_train_y = parking_price_train['parking_price'].values





# --------



parking_price_train_x_scaled = preprocessing.StandardScaler().fit_transform(parking_price_train_x)

parking_price_train_y_scaled = preprocessing.StandardScaler().fit_transform(parking_price_train_y.reshape(-1,1))

parking_price_train_X_train,parking_price_train_X_test, parking_price_train_y_train, parking_price_train_y_test = train_test_split(parking_price_train_x_scaled, parking_price_train_y_scaled, test_size=0.33, random_state=42)



# parking_price_train_clfs = {

#         'svm':svm.SVR(), #4

#         'RandomForestRegressor':RandomForestRegressor(n_estimators=400), #3

#         'BayesianRidge':linear_model.BayesianRidge(), #1

#         'xgboost' :xgboost.XGBRegressor(learning_rate=0.01,n_estimators=3460,

#                                      max_depth=3, min_child_weight=0,

#                                      gamma=0, subsample=0.7,

#                                      colsample_bytree=0.7,

#                                      objective='reg:linear', nthread=-1,

#                                      scale_pos_weight=1, seed=27,

#                                      reg_alpha=0.00006),

#         'lgb':lgb.LGBMRegressor(objective='regression', 

#                                        num_leaves=20, 

#                                        learning_rate=0.05, 

#                                        n_estimators=5000,

#                                        max_bin=200, 

#                                        bagging_fraction=0.75,

#                                        bagging_freq=5, 

#                                        bagging_seed=7,

#                                        feature_fraction=0.2, 

#                                        feature_fraction_seed=7,

#                                        verbose=-1,

#                                        num_thread=6,) #2

#        }

# for clf in parking_price_train_clfs:

#     try:

#         parking_price_train_clfs[clf].fit(parking_price_train_X_train, parking_price_train_y_train)

#         parking_price_train_y_pred = parking_price_train_clfs[clf].predict(parking_price_train_X_test)

#         print(clf + " cost:" + str(np.sum(parking_price_train_y_pred-parking_price_train_y_test)/len(parking_price_train_y_pred)) )

#     except Exception as e:

#         print(clf + " Error:")

#         print(str(e))



# --------

# parking_price_train_X_train,parking_price_train_X_test, parking_price_train_y_train, parking_price_train_y_test = train_test_split(parking_price_train_x, parking_price_train_y, test_size=0.33, random_state=42)

parking_price_train_X_train,parking_price_train_X_test, parking_price_train_y_train, parking_price_train_y_test = train_test_split(parking_price_train_x_scaled, parking_price_train_y, test_size=0.33, random_state=42)

parking_price_train_clt = linear_model.BayesianRidge()

parking_price_train_clt.fit(parking_price_train_X_train, parking_price_train_y_train)



parking_price_has_na = data_train2.loc[pd.isna(data_train2['parking_price'])]

parking_price_has_na.drop(['parking_price'], 1, inplace=True)

parking_price_has_na_X = parking_price_has_na[parking_price_train_cols].values

print('parking_price predicting...')

parking_price_has_na_Y = parking_price_train_clt.predict(parking_price_has_na_X)

print('OK')



parking_price_prediction = pd.DataFrame(parking_price_has_na_Y, columns=['parking_price'])

parking_price_has_na = parking_price_has_na.reset_index()

parking_price_result = pd.concat([parking_price_has_na, parking_price_prediction],axis = 1)

parking_price_result = parking_price_result.set_index('building_id')



# parking_price_result
# 補 txn_floor

txn_floor_train = data_train2.loc[~pd.isna(data_train2['txn_floor'])]

txn_floor_train



txn_floor_train_cols = ['total_floor','N_50','building_material','building_complete_dt','XIII_index_1000','XIII_index_50','N_500','N_1000','VII_index_50','VIII_250']

# txn_floor_train_cols = ['total_floor','N_50','building_material','building_complete_dt','XIII_index_1000']



txn_floor_train_x = txn_floor_train[txn_floor_train_cols].values

txn_floor_train_y = txn_floor_train['txn_floor'].values



# --------



txn_floor_train_x_scaled = preprocessing.StandardScaler().fit_transform(txn_floor_train_x)

txn_floor_train_y_scaled = preprocessing.StandardScaler().fit_transform(txn_floor_train_y.reshape(-1,1))

txn_floor_train_X_train, txn_floor_train_X_test, txn_floor_train_y_train, txn_floor_train_y_test = train_test_split(txn_floor_train_x_scaled, txn_floor_train_y_scaled, test_size=0.33, random_state=42)



# txn_floor_train_clfs = {

#         'svm':svm.SVR(),#5

#         'RandomForestRegressor':RandomForestRegressor(n_estimators=400), #3

#         'BayesianRidge':linear_model.BayesianRidge(), #1

#         'xgboost' :xgboost.XGBRegressor(learning_rate=0.01,n_estimators=3460,

#                                      max_depth=3, min_child_weight=0,

#                                      gamma=0, subsample=0.7,

#                                      colsample_bytree=0.7,

#                                      objective='reg:linear', nthread=-1,

#                                      scale_pos_weight=1, seed=27,

#                                      reg_alpha=0.00006),

#         'lgb':lgb.LGBMRegressor(objective='regression', 

#                                        num_leaves=20, 

#                                        learning_rate=0.05, 

#                                        n_estimators=5000,

#                                        max_bin=200, 

#                                        bagging_fraction=0.75,

#                                        bagging_freq=5, 

#                                        bagging_seed=7,

#                                        feature_fraction=0.2, 

#                                        feature_fraction_seed=7,

#                                        verbose=-1,

#                                        num_thread=6,) #2

#        }

# for clf in txn_floor_train_clfs:

#     try:

#         txn_floor_train_clfs[clf].fit(txn_floor_train_X_train, txn_floor_train_y_train)

#         txn_floor_train_y_pred = txn_floor_train_clfs[clf].predict(txn_floor_train_X_test)

#         print(clf + " cost:" + str(np.sum(txn_floor_train_y_pred-txn_floor_train_y_test)/len(txn_floor_train_y_pred)) )

#     except Exception as e:

#         print(clf + " Error:")

#         print(str(e))



# --------
# txn_floor_train_X_train, txn_floor_train_X_test, txn_floor_train_y_train, txn_floor_train_y_test = train_test_split(txn_floor_train_x, txn_floor_train_y, test_size=0.33, random_state=42)

txn_floor_train_X_train, txn_floor_train_X_test, txn_floor_train_y_train, txn_floor_train_y_test = train_test_split(txn_floor_train_x_scaled, txn_floor_train_y, test_size=0.33, random_state=42)

txn_floor_train_clt = linear_model.BayesianRidge()

txn_floor_train_clt.fit(txn_floor_train_X_train, txn_floor_train_y_train)



txn_floor_has_na = data_train2.loc[pd.isna(data_train2['txn_floor'])]

txn_floor_has_na.drop(['txn_floor'], 1, inplace=True)

txn_floor_has_na_X = txn_floor_has_na[txn_floor_train_cols].values

print('txn_floor predicting...')

txn_floor_has_na_Y = txn_floor_train_clt.predict(txn_floor_has_na_X)

print('OK')



txn_floor_prediction = pd.DataFrame(txn_floor_has_na_Y, columns=['txn_floor'])

txn_floor_has_na = txn_floor_has_na.reset_index()

txn_floor_result = pd.concat([txn_floor_has_na, txn_floor_prediction],axis = 1)

txn_floor_result = txn_floor_result.set_index('building_id')



# txn_floor_result

# 補 village_income_median

village_income_median_train = data_train2.loc[~pd.isna(data_train2['village_income_median'])]

village_income_median_train



village_income_median_train_cols = ['master_rate','doc_rate','bachelor_rate','jobschool_rate','total_floor','XIII_5000','born_rate','II_1000','lon','VII_1000']

# village_income_median_train_cols = ['master_rate','doc_rate','bachelor_rate','jobschool_rate','total_floor']



village_income_median_train_x = village_income_median_train[village_income_median_train_cols].values

village_income_median_train_y = village_income_median_train['village_income_median'].values

# -------

village_income_median_train_x_scaled = preprocessing.StandardScaler().fit_transform(village_income_median_train_x)

village_income_median_train_y_scaled = preprocessing.StandardScaler().fit_transform(village_income_median_train_y.reshape(-1,1))

village_income_median_train_X_train,village_income_median_train_X_test, village_income_median_train_y_train, village_income_median_train_y_test = train_test_split(village_income_median_train_x_scaled, village_income_median_train_y_scaled, test_size=0.33, random_state=42)



# village_income_median_train_clfs = {

#         'svm':svm.SVR(), #4

#         'RandomForestRegressor':RandomForestRegressor(n_estimators=400), #2

#         'BayesianRidge':linear_model.BayesianRidge(), #1

#         'xgboost' :xgboost.XGBRegressor(learning_rate=0.01,n_estimators=3460,

#                                      max_depth=3, min_child_weight=0,

#                                      gamma=0, subsample=0.7,

#                                      colsample_bytree=0.7,

#                                      objective='reg:linear', nthread=-1,

#                                      scale_pos_weight=1, seed=27,

#                                      reg_alpha=0.00006),

#         'lgb':lgb.LGBMRegressor(objective='regression', 

#                                        num_leaves=20, 

#                                        learning_rate=0.05, 

#                                        n_estimators=5000,

#                                        max_bin=200, 

#                                        bagging_fraction=0.75,

#                                        bagging_freq=5, 

#                                        bagging_seed=7,

#                                        feature_fraction=0.2, 

#                                        feature_fraction_seed=7,

#                                        verbose=-1,

#                                        num_thread=6,) #2

#        }

# for clf in village_income_median_train_clfs:

#     try:

#         village_income_median_train_clfs[clf].fit(village_income_median_train_X_train, village_income_median_train_y_train)

#         village_income_median_train_y_pred = village_income_median_train_clfs[clf].predict(village_income_median_train_X_test)

#         print(clf + " cost:" + str(np.sum(village_income_median_train_y_pred-village_income_median_train_y_test)/len(village_income_median_train_y_pred)) )

#     except Exception as e:

#         print(clf + " Error:")

#         print(str(e))

        

# --------
# village_income_median_train_X_train,village_income_median_train_X_test, village_income_median_train_y_train, village_income_median_train_y_test = train_test_split(village_income_median_train_x, village_income_median_train_y, test_size=0.33, random_state=42)

village_income_median_train_X_train,village_income_median_train_X_test, village_income_median_train_y_train, village_income_median_train_y_test = train_test_split(village_income_median_train_x_scaled, village_income_median_train_y, test_size=0.33, random_state=42)

village_income_median_train_clt = RandomForestRegressor(n_estimators=400)

village_income_median_train_clt.fit(village_income_median_train_X_train, village_income_median_train_y_train)





village_income_median_has_na = data_train2.loc[pd.isna(data_train2['village_income_median'])]

village_income_median_has_na.drop(['village_income_median'], 1, inplace=True)



village_income_median_has_na_X = village_income_median_has_na[village_income_median_train_cols].values

print('txn_floor predicting...')

village_income_median_has_na_Y = village_income_median_train_clt.predict(village_income_median_has_na_X)

print('OK')



village_income_median_prediction = pd.DataFrame(village_income_median_has_na_Y, columns=['village_income_median'])

village_income_median_has_na = village_income_median_has_na.reset_index()

village_income_median_result = pd.concat([village_income_median_has_na, village_income_median_prediction],axis = 1)

village_income_median_result = village_income_median_result.set_index('building_id')



# village_income_median_result
# df = pd.concat([df1, df2[~df2.index.isin(df1.index)]])

# df.update(df2)

# pd.concat([df1[~df1.index.isin(df2.index)], df2])





data_train3 = pd.read_csv('../input/train.csv')

data_train3 = data_train3.set_index('building_id')

counter = 0

for index, value in data_train3.iterrows():

    if pd.isna(value['village_income_median']):

        data_train3.loc[index, 'village_income_median'] = village_income_median_result.loc[index, 'village_income_median']

    if pd.isna(value['txn_floor']):

        data_train3.loc[index, 'txn_floor'] = txn_floor_result.loc[index, 'txn_floor']

    if pd.isna(value['parking_price']):

        data_train3.loc[index, 'parking_price'] = parking_price_result.loc[index, 'parking_price']

    if pd.isna(value['parking_area']):

        data_train3.loc[index, 'parking_area'] = parking_area_result.loc[index, 'parking_area']

    counter += 1

    print(counter,end='\r')

data_train3.to_csv('./train_nonNan.csv',index=True)
# corrmat = data_train3.corr()
# k  = 20 # 关系矩阵中将显示10个特征

# cols = corrmat.nlargest(k, 'total_price')['total_price'].index

# cm = np.corrcoef(data_train3[cols.values.T)

# sns.set(font_scale=1.25)

# hm = sns.heatmap(cm, cbar=True, annot=True, \

#                  square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)

# plt.show()
from sklearn import preprocessing

from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import numpy as np
cols = ['building_area','land_area', 'parking_area', 'master_rate', 'bachelor_rate']

# ['building_area','parking_price', 'parking_area','land_area', 'master_rate', 'bachelor_rate', 'doc_rate', 'XIII_5000','jobschool_rate', 'village_income_median','VII_5000','V_5000', 'VII_1000', 'V_1000']

# cols = ['building_area', 'land_area', 'parking_area',

#        'master_rate', 'bachelor_rate', 'doc_rate']



x = data_train3[cols].values

y = data_train3['total_price'].values

x_scaled = preprocessing.StandardScaler().fit_transform(x)

# y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))

X_train,X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# X_train,X_test, y_train, y_test = train_test_split(, y, test_size=0.33, random_state=42)
max_price = max(y_test)

print('    max_price: ' + str(max_price))

min_price = min(y_test)

print('    min_price: ' + str(min_price))

print('-----------------------')

clfs = {

#         'svm':svm.SVR(), 

        'RandomForestRegressor':RandomForestRegressor(n_estimators=400),

#         'BayesianRidge':linear_model.BayesianRidge(),

#         'xgboost' :xgboost.XGBRegressor(learning_rate=0.01,n_estimators=3460,

#                                      max_depth=3, min_child_weight=0,

#                                      gamma=0, subsample=0.7,

#                                      colsample_bytree=0.7,

#                                      objective='reg:linear', nthread=-1,

#                                      scale_pos_weight=1, seed=27,

#                                      reg_alpha=0.00006),

#         'lgb':lgb.LGBMRegressor(objective='regression', 

#                                        num_leaves=4,

#                                        learning_rate=0.01, 

#                                        n_estimators=5000,

#                                        max_bin=200, 

#                                        bagging_fraction=0.75,

#                                        bagging_freq=5, 

#                                        bagging_seed=7,

#                                        feature_fraction=0.2,

#                                        feature_fraction_seed=7,

#                                        verbose=-1,) #2

       }

for clf in clfs:

    try:

        clfs[clf].fit(X_train, y_train)

        y_pred = clfs[clf].predict(X_test)

        print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )

        max_price = max(y_pred)

        print('    max_price: ' + str(max_price))

        min_price = min(y_pred)

        print('    min_price: ' + str(min_price))

    except Exception as e:

        print(clf + " Error:")

        print(str(e))
final_clt = clfs['RandomForestRegressor']

# final_clt_rdf = RandomForestRegressor(n_estimators=400)

# print('training...')

# final_clt.fit(X_train, y_train)

# print('OK.')



# final_clt_lgb = lgb.LGBMRegressor(objective='regression', 

#                                        num_leaves=4,

#                                        learning_rate=0.01, 

#                                        n_estimators=5000,

#                                        max_bin=200, 

#                                        bagging_fraction=0.75,

#                                        bagging_freq=5, 

#                                        bagging_seed=7,

#                                        feature_fraction=0.2,

#                                        feature_fraction_seed=7,

#                                        verbose=-1,) #2

# print('training...')

# final_clt_lgb.fit(X_train, y_train)

# print('OK.')
# y1_temp = final_clt.predict(X_test)

# # y2_temp = final_clt_lgb.predict(X_test)



# # y_temp1 = (y1_temp * 0.3) + (y2_temp * 0.7)

# print(" cost:" + str(np.sum(y1_temp-y_test)/len(y1_temp)) )

# max_price = max(y1_temp)

# print('    max_price: ' + str(max_price))

# min_price = min(y1_temp)

# print('    min_price: ' + str(min_price))
data_train_test = pd.read_csv('../input/test.csv')
final_village_income_median_has_na = data_train_test.loc[pd.isna(data_train_test['village_income_median'])]

final_village_income_median_has_na.drop(['village_income_median'], 1, inplace=True)

final_village_income_median_has_na_X = final_village_income_median_has_na[village_income_median_train_cols].values

print('txn_floor predicting...')

final_village_income_median_has_na_Y = village_income_median_train_clt.predict(final_village_income_median_has_na_X)

print('OK')

final_village_income_median_prediction = pd.DataFrame(final_village_income_median_has_na_Y, columns=['village_income_median'])

final_village_income_median_has_na = final_village_income_median_has_na.reset_index()

final_village_income_median_result = pd.concat([final_village_income_median_has_na, final_village_income_median_prediction],axis = 1)

final_village_income_median_result = final_village_income_median_result.set_index('building_id')





final_txn_floor_has_na = data_train_test.loc[pd.isna(data_train_test['txn_floor'])]

final_txn_floor_has_na.drop(['txn_floor'], 1, inplace=True)

final_txn_floor_has_na_X = final_txn_floor_has_na[txn_floor_train_cols].values

print('txn_floor predicting...')

final_txn_floor_has_na_Y = txn_floor_train_clt.predict(final_txn_floor_has_na_X)

print('OK')

final_txn_floor_prediction = pd.DataFrame(final_txn_floor_has_na_Y, columns=['txn_floor'])

final_txn_floor_has_na = final_txn_floor_has_na.reset_index()

final_txn_floor_result = pd.concat([final_txn_floor_has_na, final_txn_floor_prediction],axis = 1)

final_txn_floor_result = final_txn_floor_result.set_index('building_id')





final_parking_price_has_na = data_train_test.loc[pd.isna(data_train_test['parking_price'])]

final_parking_price_has_na.drop(['parking_price'], 1, inplace=True)

final_parking_price_has_na_X = final_parking_price_has_na[parking_price_train_cols].values

print('parking_price predicting...')

final_parking_price_has_na_Y = parking_price_train_clt.predict(final_parking_price_has_na_X)

print('OK')

final_parking_price_prediction = pd.DataFrame(final_parking_price_has_na_Y, columns=['parking_price'])

final_parking_price_has_na = final_parking_price_has_na.reset_index()

final_parking_price_result = pd.concat([final_parking_price_has_na, final_parking_price_prediction],axis = 1)

final_parking_price_result = final_parking_price_result.set_index('building_id')





final_parking_area_has_na = data_train_test.loc[pd.isna(data_train_test['parking_area'])]

final_parking_area_has_na.drop(['parking_area'], 1, inplace=True)

final_parking_area_has_na_X = final_parking_area_has_na[parking_area_train_cols].values

print('parking_area predicting...')

final_parking_area_has_na_Y = parking_area_train_clt.predict(final_parking_area_has_na_X)

print('OK')

final_parking_area_prediction = pd.DataFrame(final_parking_area_has_na_Y, columns=['parking_area'])

final_parking_area_has_na = final_parking_area_has_na.reset_index()

final_parking_area_result = pd.concat([final_parking_area_has_na, final_parking_area_prediction],axis = 1)

final_parking_area_result = final_parking_area_result.set_index('building_id')
data_train_test = data_train_test.set_index('building_id')

counter = 0

for index, value in data_train_test.iterrows():

    if pd.isna(value['village_income_median']):

        data_train_test.loc[index, 'village_income_median'] = final_village_income_median_result.loc[index, 'village_income_median']

    if pd.isna(value['txn_floor']):

        data_train_test.loc[index, 'txn_floor'] = final_txn_floor_result.loc[index, 'txn_floor']

    if pd.isna(value['parking_price']):

        data_train_test.loc[index, 'parking_price'] = final_parking_price_result.loc[index, 'parking_price']

    if pd.isna(value['parking_area']):

        data_train_test.loc[index, 'parking_area'] = final_parking_area_result.loc[index, 'parking_area']

    counter += 1

    print(counter,end='\r')

data_train_test.to_csv('./test_nonNan.csv',index=True)
# cols = ['building_area','land_area', 'parking_area', 'master_rate', 'bachelor_rate']



data_train_test_X = data_train_test[cols].values

data_train_test_X_scaled = preprocessing.StandardScaler().fit_transform(data_train_test_X)



# data_train_test_Y = final_clt.predict(data_train_test_X)

data_train_test_Y = final_clt.predict(data_train_test_X_scaled)





data_train_test_prediction = pd.DataFrame(data_train_test_Y, columns=['total_price'])

data_train_test2 = data_train_test.reset_index()

final_parking_area_result = pd.concat([data_train_test2, data_train_test_prediction],axis = 1)

final_parking_area_result = final_parking_area_result.set_index('building_id')

# final_parking_area_result = final_parking_area_result['total_price']

# final_parking_area_result

max_price = max(final_parking_area_result['total_price'].values)

print('max_price: ' + str(max_price))

min_price = min(final_parking_area_result['total_price'].values)

print('min_price: ' + str(min_price))

final_parking_area_result = final_parking_area_result[['total_price']]

final_parking_area_result

final_parking_area_result.to_csv('./answer.csv',index=True)
!ls