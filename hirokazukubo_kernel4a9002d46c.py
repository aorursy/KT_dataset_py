# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_columns', 200)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/exam-for-students20200527/train.csv")

station_info_df = pd.read_csv("/kaggle/input/exam-for-students20200527/station_info.csv")

data_dictionary_df = pd.read_csv("/kaggle/input/exam-for-students20200527/data_dictionary.csv")

test_df = pd.read_csv("/kaggle/input/exam-for-students20200527/test.csv")

city_info_df = pd.read_csv("/kaggle/input/exam-for-students20200527/city_info.csv")

sample_submission_df = pd.read_csv("/kaggle/input/exam-for-students20200527/sample_submission.csv")
train_df.head()
print(train_df.shape)
train_df.describe()
data_dictionary_df
# test_dfの確認

test_df.head()
test_df.shape
# station_info_dfの確認

station_info_df.head()
# city_info_dfの確認

city_info_df.head()
# sample_submission_dfの確認

sample_submission_df.head()
train_df.describe()
test_df.describe()
train_df.isnull().sum()
test_df.isnull().sum()
# trainとtestをマージ最終的にはid251487以降でテストに分割

all_df = pd.concat([train_df,test_df])

all_df.head()
all_df.shape
city_info_df.columns = ['Prefecture', 'Municipality', 'City_Latitude', 'City_Longitude']

station_info_df.columns = ['NearestStation', 'Station_Latitude', 'Station_Longitude']
all_df_c = pd.merge(all_df,city_info_df,on=["Prefecture","Municipality"],how="left")

all_df_cs = pd.merge(all_df_c,station_info_df,on="NearestStation",how="left")

all_df_cs.head()
all_df_cs.isnull().sum()
all_df_cs.groupby("Type").mean()
all_df_cs[all_df_cs["Remarks"].isnull()==False]["Remarks"].value_counts()
train_df[train_df["TradePrice"]<50000]["TradePrice"].hist(bins=50)
# 目的変数はlogとる

train_df["TradePrice"].apply(np.log).hist()
# いったん全部ordinal

import category_encoders as ce



# Eoncodeしたい列をリストで指定。

list_cols = ['Type',"Region","FloorPlan","LandShape","FrontageIsGreaterFlag","Structure","Use","Purpose","Direction","Classification","CityPlanning","Renovation","Remarks"]



# 序数をカテゴリに付与して変換

ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')

df_session_ce_ordinal = ce_oe.fit_transform(all_df_cs)



df_session_ce_ordinal.head()

df_session_ce_ordinal.columns
df_focus = df_session_ce_ordinal[['id', 'Type', 'Region', 'MinTimeToNearestStation',

       'MaxTimeToNearestStation', 'FloorPlan', 'Area', 'AreaIsGreaterFlag',

       'LandShape', 'Frontage', 'FrontageIsGreaterFlag', 'TotalFloorArea',

       'TotalFloorAreaIsGreaterFlag', 'BuildingYear', 'PrewarBuilding',

       'Structure', 'Use', 'Purpose', 'Direction', 'Classification', 'Breadth',

       'CityPlanning', 'CoverageRatio', 'FloorAreaRatio', 'Year', 'Quarter',

       'Renovation', 'Remarks', 'City_Latitude',

       'City_Longitude', 'Station_Latitude', 'Station_Longitude', 'TradePrice']]
train = df_focus[df_focus["id"]<251487]

test = df_focus[df_focus["id"]>=251487]
train.head()
test.head()
# import optuna

# import lightgbm



# X = train.iloc[:,:32]

# y = np.log(train["TradePrice"])

# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 144) 



# def opt(trial):

# #     n_estimators = trial.suggest_int('n_estimators', 10000)

#     max_depth = trial.suggest_int('max_depth', 5, 10)

# #     max_leaves = trial.suggest_int('max_leaves',max_depth^2)

#     min_child_weight = trial.suggest_int('min_child_weight', 10, 50)

# #     subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)

# #     colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

#     model_opt = lightgbm.LGBMRegressor(

#         random_state=42,

#         n_estimators = 15000,

#         max_depth = max_depth,

#         max_leaves = max_depth^2,

#         min_child_weight = min_child_weight,

# #         subsample = subsample,

# #         colsample_bytree = colsample_bytree

#     )

# #     gbm.fit(X_train, y_train,

# #         eval_set=[(X_test, y_test)],

# #         eval_metric='multi_logloss',

# #         early_stopping_rounds=10)

#     model_opt.fit(train_X,train_y

#                   ,eval_set=[(test_X, test_y)]

# #                   ,eval_metric='multi_logloss',

#                   ,early_stopping_rounds = 100)

#     opt_pred = model_opt.predict(test_X)

#     return (1.0 - (model_opt.score(test_X, test_y)))

# model_opt=lightgbm.LGBMRegressor()

# study = optuna.create_study()

# study.optimize(opt, n_trials=5)

# print(study.best_params)

# print(1-study.best_value)
import lightgbm

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

# 元の値はmax_depth=7,min_child_weight = 16



X = train.iloc[:,:32]

y = np.log(train["TradePrice"])

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 144) 



# X_trainval, X_test, y_trainval, y_test = train_test_split(boston.data, boston.target, random_state=0)

# X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=0)



def rmsle(y, y0):

     assert len(y) == len(y0)

     return np.sqrt(np.mean(np.power(np.log(y)-np.log(y0), 2)))



lgb = lightgbm.LGBMRegressor(n_estimators=10000,

                             max_depth=6,

                             max_leaves = 36, # max_depth^2

                             min_child_weight = 47,

                             lambda_l1 = 3.0,

                             lambda_l2 = 3.0,

#                               lambda_l1 = 0.5,

#                              lambda_l2 = 0.6,

#                             subsample=0.9,

#                             colsample_bytree=0.9,

                            random_state=42)

lgb.fit(train_X, train_y

        ,eval_set=[(test_X, test_y)]

        ,early_stopping_rounds = 100)

Y_train_lgb = lgb.predict(train_X)

Y_test_lgb = lgb.predict(test_X)



print('-----lightGBM-----')

print('MSE train data: ', np.sqrt(mean_squared_error(np.exp(train_y), np.exp(Y_train_lgb)))) # 学習データを用いたときの平均二乗誤差を出力

print('MSE test data: ', np.sqrt(mean_squared_error(np.exp(test_y), np.exp(Y_test_lgb))))
print ("train RMSLE = " + str(rmsle(np.exp(train_y), np.exp(Y_train_lgb))))

print ("test RMSLE = " + str(rmsle(np.exp(test_y), np.exp(Y_test_lgb))))
pd.concat([np.exp(pd.DataFrame(train_y)).reset_index(),np.exp(pd.DataFrame(Y_train_lgb))],axis=1)
pred_submit = np.exp(lgb.predict(test.iloc[:,:32]))
submission = pd.concat([test["id"].reset_index(),pd.DataFrame(pred_submit)],axis=1).iloc[:,1:]

submission.head()
submission.columns = ["id","TradePrice"]

submission.to_csv('submision.csv',index=False)