def reduce_mem_usage(props):

    start_mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if props[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            if props[col].dtype=='datetime64[ns]':

                continue 

            print("dtype before: ",props[col].dtype)

            

            

            # make variables for Int, max and min

            IsInt = False

            mx = props[col].max()

            mn = props[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(props[col]).all(): 

                NAlist.append(col)

                props[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = props[col].fillna(0).astype(np.int64)

            result = (props[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        props[col] = props[col].astype(np.uint8)

                    elif mx < 65535:

                        props[col] = props[col].astype(np.uint16)

                    elif mx < 4294967295:

                        props[col] = props[col].astype(np.uint32)

                    else:

                        props[col] = props[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        props[col] = props[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        props[col] = props[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        props[col] = props[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        props[col] = props[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                props[col] = props[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",props[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return props, NAlist
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





#submission = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv", parse_dates=["Date"])

train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv", parse_dates=["Date"])

global_data = pd.read_csv("../input/externalcountrydata/Global_Data_by_Country_2019.csv")



reduce_mem_usage(test)

reduce_mem_usage(train)

reduce_mem_usage(global_data)

global_data.head()

global_data.groupby("CountryName")["ExtraColumn"].sum()
train.loc[train["Province/State"].isnull(), "Province/State"]=train.loc[train["Province/State"].isnull(), "Country/Region"]



train.rename(columns = {'Country/Region':'Country', 'Province/State':'Province'}, inplace = True) 







train=train.merge(global_data, how='left', left_on=['Country', 'Province'], right_on=['CountryName', 'Province'])

train.head()
test.rename(columns = {'Country/Region':'Country', 'Province/State':'Province'}, inplace = True) 



test.loc[test["Province"].isnull(), "Province"]=test.loc[test["Province"].isnull(), "Country"]



X_test=test.merge(global_data, how='left', left_on=['Country', 'Province'], right_on=['CountryName', 'Province'])

X_test=X_test.drop("Population", axis=1)

X_test=X_test.drop("CountryName", axis=1)

X_test=X_test.rename(columns={"ExtraColumn": "Population"})

del global_data
train=train.drop("Population", axis=1)

train=train.drop("CountryName", axis=1)



train=train.rename(columns={"ExtraColumn": "Population"})

mindates = train[train["ConfirmedCases"]>0].groupby(['Province'])["Date"].min()

mindates.reset_index()

mindatesDF = mindates.to_frame()

mindatesDF.rename(columns={"Date":"MinDate"}, inplace=True)

train=train.merge(mindatesDF, how='left', left_on="Province", right_on="Province")

train["DaysFrom1stCase"]=(train["Date"]-train["MinDate"]).dt.days

train.loc [train["DaysFrom1stCase"]<0 , "DaysFrom1stCase"] =0



## after version 2 ###

#train=pd.get_dummies(train, prefix='prov', prefix_sep='_', dummy_na=True, columns="Province", sparse=False, drop_first=False, dtype=None)



from sklearn.preprocessing import LabelEncoder 



labelencoder = LabelEncoder()

province=labelencoder.fit_transform(train["Province"])

train=pd.concat([train, pd.DataFrame(province)], axis=1)


train=train.drop(["Country","MinDate", "Province"], axis=1)

#train=train.drop(["Country","MinDate"], axis=1)
train[train.ConfirmedCases>0].head(100)
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA





y_train_CC=train.loc[:,"ConfirmedCases"]

y_train_F=train.loc[:, "Fatalities"]

X_train=train.drop(["ConfirmedCases", "Fatalities", "Id", "Date"], axis=1)
X_train.head()
for col in X_train.columns:

    X_train[col]=X_train[col].fillna(X_train[col].median())
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



#pca = PCA(n_components=5)

#pca.fit(X_train)



#X_train= pd.DataFrame(pca.transform(X_train))





X_train_real, X_test_val, y_train_real_CC, y_test_val_CC = train_test_split(

        X_train, y_train_CC, test_size=0.3, random_state=0)







lgb_train_CC = lgb.Dataset(X_train_real, y_train_real_CC)

lgb_eval_CC = lgb.Dataset(X_test_val, y_test_val_CC, reference=lgb_train_CC)





X_train_real, X_test_val, y_train_real_F, y_test_val_F = train_test_split(

        X_train, y_train_F, test_size=0.3, random_state=0)



lgb_train_F = lgb.Dataset(X_train_real, y_train_real_F)

lgb_eval_F = lgb.Dataset(X_test_val, y_test_val_F, reference=lgb_train_F)





#X_train.head()
X_train_real.head()
params = {

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



gbm_CC = lgb.train(params,

            lgb_train_CC,

            num_boost_round=100,

            valid_sets=lgb_eval_CC,

            early_stopping_rounds=10)



gbm_F = lgb.train(params,

            lgb_train_F,

            num_boost_round=100,

            valid_sets=lgb_eval_F,

            early_stopping_rounds=10)


test.head()


## add days since first case column ##

X_test=X_test.merge(mindatesDF, how='left', left_on="Province", right_on="Province")

X_test["DaysFrom1stCase"]=(X_test["Date"]-X_test["MinDate"]).dt.days

X_test.loc [X_test["DaysFrom1stCase"]<0 , "DaysFrom1stCase"] =0



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

y_pred_CC = gbm_CC.predict(X_test, num_iteration=gbm_CC.best_iteration)

y_pred_F = gbm_F.predict(X_test, num_iteration=gbm_F.best_iteration)

y_pred.shape
forecastId=test.ForecastId.to_numpy()

submission_CC=pd.DataFrame(y_pred_CC)

submission_F=pd.DataFrame(y_pred_F)

forecastIdDF=pd.DataFrame(forecastId)

forecastIdDF=forecastIdDF.rename(columns={0:"ForecastId"})

submission=pd.concat([forecastIdDF, submission_CC, submission_F ], axis=1)

submission=submission.rename(columns={0:"ConfirmedCases", 1:"Fatalities"})
submission.head()
submission.to_csv("submission.csv", index=False)
test=pd.concat([test, submission ], axis=1)
test[test.Province=="Italy"]
test.ConfirmedCases.sum()


