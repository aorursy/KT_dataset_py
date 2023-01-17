import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import matplotlib.pyplot as plt

%matplotlib inline



from collections import Counter

import category_encoders as ce



from sklearn import preprocessing





import warnings

warnings.filterwarnings("ignore")



#about file

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
result_example = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

result_example.head(),result_example.shape
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

train.head(3)
test.head(3)
train.shape
train_new = train.groupby(["Country_Region","Date"],as_index= False).sum().drop('Id',axis = 1)

train_new.shape
train["Date"] = pd.to_datetime(train["Date"]).dt.strftime("%m%d").astype(int) 

train['Date'] -= 122 #because first_date = 122

test["Date"] = pd.to_datetime(test["Date"]).dt.strftime("%m%d").astype(int) 

test['Date'] -= 122
train['Province_State'].fillna(train['Country_Region'],inplace = True)

test['Province_State'].fillna(test['Country_Region'],inplace = True)
train.Country_Region
le = preprocessing.LabelEncoder()



train.Country_Region = le.fit_transform(train.Country_Region)

train.Province_State = le.fit_transform(train.Province_State)

test.Country_Region = le.fit_transform(test.Country_Region)

test.Province_State = le.fit_transform(test.Province_State)



#scaling did not improve score



#train.Country_Region = preprocessing.scale(train.Country_Region)

#train.Province_State = preprocessing.scale(train.Province_State)

#test.Country_Region = preprocessing.scale(test.Country_Region)

#test.Province_State = preprocessing.scale(test.Province_State)
train.tail(3), test.head(3)
#open_period only_use data 1/22~3/19

val = train[train["Date"] > 196].reset_index(drop = True)

train = train[train["Date"] <= 196].reset_index(drop = True)



#scaling Date gave bad socre!



#val.Date= preprocessing.scale(val.Date)

#train.Date= preprocessing.scale(train.Date)

#test.Date= preprocessing.scale(test.Date)
x_train = train.drop(['Id', 'ConfirmedCases','Fatalities'], axis = 1)

y_train_1 = train["ConfirmedCases"]

y_train_2 = train["Fatalities"]

x_val = val.drop(['Id', 'ConfirmedCases','Fatalities'], axis = 1)

y_val_1 = val["ConfirmedCases"]

y_val_2 = val["Fatalities"]

train.shape , val.shape
val.tail()
import lightgbm as lgb

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor
def evaluate(predict,actual):

    #eval = 0

    #for i in range(n):

        #eval += float(np.square( np.log(predict[i]+1) - np.log(actual[i]+1) ) )                                                

    return np.sqrt(np.mean(np.square(np.log(predict+1) - np.log(actual+1))))
lgbm_params = {

   'objective': 'regression',

    'metric': 'rmse',

}

lgb_train_1 = lgb.Dataset(x_train, y_train_1)

lgb_eval_1 = lgb.Dataset(x_val, y_val_1, reference=lgb_train_1)



lgb_model_1 = lgb.train(lgbm_params,lgb_train_1, valid_sets=lgb_eval_1,num_boost_round=10000)

y_pred_1 = lgb_model_1.predict(x_val, num_iteration = lgb_model_1.best_iteration)



lgb_train_2 = lgb.Dataset(x_train, y_train_2)

lgb_eval_2 = lgb.Dataset(x_val, y_val_2, reference=lgb_train_2)



lgb_model_2 = lgb.train(lgbm_params,lgb_train_2, valid_sets=lgb_eval_2,num_boost_round=5000)

y_pred_2 = lgb_model_1.predict(x_val, num_iteration = lgb_model_2.best_iteration)



evaluate(y_pred_1,y_val_1), evaluate(y_pred_2,y_val_2),evaluate(y_pred_1,y_val_1)+evaluate(y_pred_2,y_val_2)

lgb_model_3= lgb.LGBMRegressor(n_estimators=3000)

lgb_model_3.fit(x_train,y_train_1)

y_pred_1 = lgb_model_3.predict(x_val)

lgb_model_4= lgb.LGBMRegressor(n_estimators=3000)

lgb_model_4.fit(x_train,y_train_2)

y_pred_2 = lgb_model_4.predict(x_val)



evaluate(y_pred_1,y_val_1), evaluate(y_pred_2,y_val_2),evaluate(y_pred_1,y_val_1)+evaluate(y_pred_2,y_val_2)
xgb_model_1= XGBRegressor(n_estimators = 4000)

xgb_model_1.fit(x_train,y_train_1)

y_pred_1 = xgb_model_1.predict(x_val)

xgb_model_2= XGBRegressor(n_estimators = 4000)

xgb_model_2.fit(x_train,y_train_2)

y_pred_2 = xgb_model_2.predict(x_val)



evaluate(y_pred_1,y_val_1), evaluate(y_pred_2,y_val_2),evaluate(y_pred_1,y_val_1) + evaluate(y_pred_2,y_val_2)
random_forest_1=RandomForestRegressor(bootstrap=True, 

            max_depth=25, max_features='auto', max_leaf_nodes=None,

            min_samples_leaf=1, min_samples_split=15,

            min_weight_fraction_leaf=0.0, n_estimators=150, 

            random_state=0, verbose=0, warm_start=False)

random_forest_1.fit(x_train,y_train_1)

random_forest_1 = xgb_model_1.predict(x_val)



random_forest_2=RandomForestRegressor(bootstrap=True, 

            max_depth=25, max_features='auto', max_leaf_nodes=None,

            min_samples_leaf=1, min_samples_split=15,

            min_weight_fraction_leaf=0.0, n_estimators=150, 

            random_state=0, verbose=0, warm_start=False)

random_forest_2.fit(x_train,y_train_2)

y_pred_2 = random_forest_2.predict(x_val)

evaluate(y_pred_1,y_val_1), evaluate(y_pred_2,y_val_2),evaluate(y_pred_1,y_val_1)+evaluate(y_pred_2,y_val_2)
x_test = test.drop(['ForecastId'],axis = 1)

ans_1 =xgb_model_1.predict(x_test)

ans_2 = xgb_model_2.predict(x_test)
test["ConfirmedCases"] = ans_1.astype(int)

test["Fatalities"] = ans_2.astype(int)

test.drop(["Date","Country_Region","Province_State"],axis = 1,inplace = True)

test.to_csv("submission.csv",index = False)