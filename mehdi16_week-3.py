# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





#submission = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv", parse_dates=["Date"])

train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv", parse_dates=["Date"])

global_data = pd.read_csv("../input/external/Global_Data_by_Country_2019.csv")

coordinates=pd.read_csv("../input/external/coordinates.csv")



coordinates.head()
coordinates=coordinates[["Country/Region", "Province/State", "Lat", "Long"]]

coordinates.loc[coordinates["Province/State"].isnull(), "Province/State"]=coordinates.loc[coordinates["Province/State"].isnull(), "Country/Region"]

coordinates.drop_duplicates(keep='first', inplace=True)





from math import radians, sin, cos, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1

    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    return 2 * 6371 * asin(sqrt(a))





coordinates.set_index("Province/State")

latitude=coordinates[coordinates["Province/State"]=="Hubei"].Lat

longitude=coordinates[coordinates["Province/State"]=="Hubei"].Long



coordinates["Distance"]=coordinates.apply(lambda x : haversine(lat1=x.Lat, lon1=x.Long, lat2=latitude, lon2=longitude), axis=1)

coordinates=coordinates[["Province/State", "Distance"]]





coordinates.head()
### when there's no province replace it by country name

train.loc[train["Province_State"].isnull(), "Province_State"]=train.loc[train["Province_State"].isnull(), "Country_Region"]

###join train data with external data about country mainly : area, population, life expectancy, health spendings

train.rename(columns = {'Country_Region':'Country', 'Province_State':'Province'}, inplace = True) 

train=train.merge(global_data, how='left', left_on=['Country', 'Province'], right_on=['CountryName', 'Province'])

train=train.merge(coordinates, how='left', left_on=['Province'], right_on=['Province/State'])

train=train.drop("Province/State", axis=1)

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

test.rename(columns = {'Country_Region':'Country', 'Province_State':'Province'}, inplace = True) 

test.loc[test["Province"].isnull(), "Province"]=test.loc[test["Province"].isnull(), "Country"]

X_test=test.merge(global_data, how='left', left_on=['Country', 'Province'], right_on=['CountryName', 'Province'])

X_test=X_test.drop("Population", axis=1)

X_test=X_test.drop("CountryName", axis=1)

X_test=X_test.rename(columns={"ExtraColumn": "Population"})

X_test=X_test.merge(coordinates, how='left', left_on=['Province'], right_on=['Province/State'])

X_test=X_test.drop("Province/State", axis=1)

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
for col in X_train.columns:

    X_train[col]=X_train[col].fillna(X_train[col].median())




from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



#pca = PCA(n_components=5)

#pca.fit(X_train)



#X_train= pd.DataFrame(pca.transform(X_train))



X_train_CC=X_train.drop(["LifeExpectancy", "Population"], axis=1)



X_train_real, X_test_val, y_train_real_CC, y_test_val_CC = train_test_split(

        X_train_CC, y_train_CC, test_size=0.3, random_state=0)







lgb_train_CC = lgb.Dataset(X_train_real, y_train_real_CC)

lgb_eval_CC = lgb.Dataset(X_test_val, y_test_val_CC, reference=lgb_train_CC)



X_train_F=X_train.drop(["LifeExpectancy", "Distance"], axis=1)



X_train_real, X_test_val, y_train_real_F, y_test_val_F = train_test_split(

        X_train_F, y_train_F, test_size=0.3, random_state=0)



lgb_train_F = lgb.Dataset(X_train_real, y_train_real_F)

lgb_eval_F = lgb.Dataset(X_test_val, y_test_val_F, reference=lgb_train_F)







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

X_test_CC=X_test.drop(["LifeExpectancy", "Population"], axis=1)

X_test_F=X_test.drop(["LifeExpectancy", "Population"], axis=1)

#y_pred=regressor.predict(X_test)

y_pred_CC = gbm_CC.predict(X_test_CC, num_iteration=gbm_CC.best_iteration)

y_pred_F = gbm_F.predict(X_test_F, num_iteration=gbm_F.best_iteration)

#forecastId=test.ForecastId.to_numpy()

submission_CC=pd.DataFrame(y_pred_CC)

submission_CC=submission_CC.rename(columns={0:"ConfirmedCases"})

submission_F=pd.DataFrame(y_pred_F)

submission_F=submission_F.rename(columns={0:"Fatalities"})

#submission["ForecastId"]=test["ForecastId"]

submission=test.copy()

submission=submission.drop(columns=["Province", "Country", "Date"])

submission["ConfirmedCases"]=submission_CC["ConfirmedCases"]

submission["Fatalities"]=submission_F["Fatalities"]

submission.head()



submission=submission.dropna()
submission.to_csv("submission.csv", index=False)
submission.head()
submission.info()
test=pd.concat([test, submission ], axis=1)
test[test["Country"]=="Algeria"]