# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





#submission = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv", parse_dates=["Date"])

train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv", parse_dates=["Date"])

global_data = pd.read_csv("../input/externalcountrydata/Global_Data_by_Country_2019.csv")

country_info=pd.read_csv("../input/countryinfo/covid19countryinfo.csv")



train.loc[train["Province_State"].isnull(), "Province_State"]=train.loc[train["Province_State"].isnull(), "Country_Region"]



train.rename(columns = {'Country_Region':'Country', 'Province_State':'Province'}, inplace = True) 







train=train.merge(global_data, how='left', left_on=['Country', 'Province'], right_on=['CountryName', 'Province'])

country_info.info()
#train=train.merge(country_info[["country", "medianage"]], how='left', left_on='Country', right_on='country' )



#medianage, smoker, hospibed, lung
train.info()
test.rename(columns = {'Country_Region':'Country', 'Province_State':'Province'}, inplace = True) 

#test=test.merge(country_info[["country", "medianage"]], how='left', left_on='Country', right_on='country' )

test.loc[test["Province"].isnull(), "Province"]=test.loc[test["Province"].isnull(), "Country"]



X_test=test.merge(global_data, how='left', left_on=['Country', 'Province'], right_on=['CountryName', 'Province'])

X_test=X_test.drop("Population", axis=1)

X_test=X_test.drop("CountryName", axis=1)

X_test=X_test.rename(columns={"ExtraColumn": "Population"})

del global_data
train=train.drop("Population", axis=1)

train=train.drop("CountryName", axis=1)



train=train.rename(columns={"ExtraColumn": "Population"})

train.head()
mindates = train[train["ConfirmedCases"]>0].groupby(['Province'])["Date"].min()

mindates.reset_index()

mindatesDF = mindates.to_frame()

mindatesDF.rename(columns={"Date":"MinDate"}, inplace=True)

train=train.merge(mindatesDF, how='left', left_on="Province", right_on="Province")

train["DaysFrom1stCase"]=(train["Date"]-train["MinDate"]).dt.days

train.loc [train["DaysFrom1stCase"]<0 , "DaysFrom1stCase"] =0

first_day=train[train["ConfirmedCases"]>0].Date.min()

train["DaysFromStart"]=(train["Date"]-first_day).dt.days 

## after version 2 ###

#train=pd.get_dummies(train, prefix='prov', prefix_sep='_', dummy_na=True, columns="Province", sparse=False, drop_first=False, dtype=None)



train.head()
from sklearn.preprocessing import LabelEncoder 



labelencoder = LabelEncoder()

province=labelencoder.fit_transform(train["Province"])

train=pd.concat([train, pd.DataFrame(province)], axis=1)


train=train.drop(["Country","MinDate", "Province"], axis=1)

#train=train.drop(["Country","MinDate"], axis=1)
train.rename(columns={0:"Province"})
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer

from sklearn.linear_model import RidgeCV

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA





y_train_CC=train.loc[:,"ConfirmedCases"]

y_train_F=train.loc[:, "Fatalities"]

X_train=train.drop(["ConfirmedCases", "Fatalities", "Id", "Date"], axis=1)
for col in X_train.columns:

    X_train[col]=X_train[col].fillna(X_train[col].median())
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



#pca = PCA(n_components=7)

#pca.fit(X_train)

#X_train= pd.DataFrame(pca.transform(X_train))







X_train_real_CC, X_test_val_CC, y_train_real_CC, y_test_val_CC = train_test_split(

        X_train, y_train_CC, test_size=0.3, random_state=0)







lgb_train_CC = lgb.Dataset(X_train_real_CC, y_train_real_CC)

lgb_eval_CC = lgb.Dataset(X_test_val_CC, y_test_val_CC, reference=lgb_train_CC)





X_train_real_F, X_test_val_F, y_train_real_F, y_test_val_F = train_test_split(

        X_train, y_train_F, test_size=0.3, random_state=0)



lgb_train_F = lgb.Dataset(X_train_real, y_train_real_F)

lgb_eval_F = lgb.Dataset(X_test_val, y_test_val_F, reference=lgb_train_F)





#X_train.head()
params_CC = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'regression',

        'metric': {'rmse'},

        'learning_rate': 0.3,

        'num_leaves': 30,

        'min_data_in_leaf': 1,

        'num_iteration': 100,

        'verbose': 20

}





params_F = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'regression',

        'metric': {'rmse'},

        'learning_rate': 0.28,

        'num_leaves': 35,

        'min_data_in_leaf': 2,

        'num_iteration': 100,

        'verbose': 20

}





gbm_CC = lgb.LGBMRegressor(n_jobs=-1)

gbm_F = lgb.LGBMRegressor(n_jobs=-1)



param_grid = {

    "num_leaves" : np.linspace(10, 200, 4, dtype=np.int32),

    'learning_rate': np.linspace(0.1, 1, 5),

    'n_estimators': np.linspace(10, 1000, 5, dtype=np.int32),

    'early_stopping_rounds' : [20],

}



gbm_CC = GridSearchCV(gbm_CC, param_grid, cv=3, scoring="neg_mean_squared_error", verbose=100, n_jobs=-1)

gbm_CC.fit(X_train_real_CC, y_train_real_CC, eval_set=[(X_test_val_CC, y_test_val_CC)], eval_metric="rmse")

print('Best parameters:', gbm_CC.best_params_)



gbm_F = GridSearchCV(gbm_F, param_grid, cv=3, scoring="neg_mean_squared_error", verbose=100, n_jobs=-1)

gbm_F.fit(X_train_real_F, y_train_real_F, eval_set=[(X_test_val_F, y_test_val_F)], eval_metric="rmse")

print('Best parameters:', gbm_F.best_params_)



test.head()


## add days since first case column ##

X_test=X_test.merge(mindatesDF, how='left', left_on="Province", right_on="Province")

X_test["DaysFrom1stCase"]=(X_test["Date"]-X_test["MinDate"]).dt.days

X_test.loc [X_test["DaysFrom1stCase"]<0 , "DaysFrom1stCase"] =0

X_test["DaysFromStart"]=(X_test["Date"]-first_day).dt.days 



#X_test=pd.get_dummies(X_test, prefix='prov', prefix_sep='_', dummy_na=True, columns="Province", sparse=False, drop_first=False, dtype=None)



X_test=X_test.drop(["Country", "MinDate"], axis=1)



X_test=X_test.drop(["ForecastId", "Date"], axis=1)









province=labelencoder.transform(X_test["Province"])

X_test=pd.concat([X_test,pd.DataFrame(province) ], axis=1)

X_test=X_test.drop(["Province"], axis=1)

for col in X_test.columns:

    X_test[col]=X_test[col].fillna(X_test[col].median())
X_test.head()
#X_test= pca.transform(X_test)



#y_pred=regressor.predict(X_test)

#y_pred_CC = gbm_CC.predict(X_test, num_iteration=gbm_CC.best_iteration)

#y_pred_F = gbm_F.predict(X_test, num_iteration=gbm_F.best_iteration)





y_pred_CC = gbm_CC.predict(X_test)

y_pred_F = gbm_F.predict(X_test)

forecastId=test.ForecastId.to_numpy()

submission_CC=pd.DataFrame(y_pred_CC)

submission_CC=submission_CC.rename(columns={0:"ConfirmedCases"})

submission_F=pd.DataFrame(y_pred_F)

submission_F=submission_F.rename(columns={0:"Fatalities"})

forecastIdDF=pd.DataFrame(forecastId)

forecastIdDF=forecastIdDF.rename(columns={0:"ForecastId"})

submission=pd.concat([forecastIdDF, submission_CC, submission_F ], axis=1)
submission.head()
submission.to_csv("submission.csv", index=False)
submission.info()
test=pd.concat([test, submission ], axis=1)
test[test.Province=="Algeria"]