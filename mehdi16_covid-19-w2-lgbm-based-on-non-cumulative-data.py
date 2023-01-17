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

global_data=global_data.drop( ["HealthExpenditure"], axis=1)

train.loc[train["Province_State"].isnull(), "Province_State"]=train.loc[train["Province_State"].isnull(), "Country_Region"]



train.rename(columns = {'Country_Region':'Country', 'Province_State':'Province'}, inplace = True) 







train=train.merge(global_data, how='left', left_on=['Country', 'Province'], right_on=['CountryName', 'Province'])

#train=train.merge(country_info[["country", "medianage","urbanpop"]], how='left', left_on='Country', right_on='country' )
test.rename(columns = {'Country_Region':'Country', 'Province_State':'Province'}, inplace = True) 



test.loc[test["Province"].isnull(), "Province"]=test.loc[test["Province"].isnull(), "Country"]

#test=test.merge(country_info[["country", "medianage", "urbanpop"]], how='left', left_on='Country', right_on='country' )





X_test=test.merge(global_data, how='left', left_on=['Country', 'Province'], right_on=['CountryName', 'Province'])

X_test=X_test.drop("Population", axis=1)

X_test=X_test.drop(["CountryName"], axis=1)

X_test=X_test.rename(columns={"ExtraColumn": "Population"})

train.head()
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



def adds_to_cumulative(df, col, newlabel):

    work=df[df.Province=="No Such Country"]

    work[newlabel]=np.NaN

    print(1)

    for province in df.Province.unique():

        subframe=df[df.Province==province].copy()

        subframe=subframe.sort_values("Date")

        minvalue=subframe[col].min()

        subframesize=df[df.Province==province].size

        temparray=subframe[col].values.tolist()

        for i in range(1,len(temparray)-1 ):

            temparray[i]=temparray[i]+temparray[i-1]

            #print(temparray[i],temparray[i]-temparray[i-1])

        temparray[0]=minvalue

        subframe[newlabel]=temparray

        #print(country, temparray[-1])

        work=pd.concat([work,subframe])

        del(subframe)

        del(temparray)

        #work.to_csv("work.csv")

    #df=df.merge(subframe[["Country", "Date", "add"]], how="left", left_on=["Country", "Date"], right_on=["Country", "Date"])

    return work 

def cumulative_to_adds(df, col, newlabel):

    work=df[df.Province=="No Such Country"]

    work[newlabel]=np.NaN

    print(1)

    for province in df.Province.unique():

        subframe=df[df.Province==province].copy()

        minvalue=subframe[col].min()

        df=df.sort_values("Date")

        subframesize=df[df.Province==province].size

        temparray=subframe[col].values.tolist()

        for i in range(len(temparray)-1,1,-1 ):

            temparray[i]=temparray[i]-temparray[i-1]

            #print(temparray[i],temparray[i]-temparray[i-1])

        temparray[0]=minvalue

        subframe[newlabel]=temparray

        #print(country, temparray[-1])

        work=pd.concat([work,subframe])

        del(subframe)

        del(temparray)

        #work.to_csv("work.csv")

    #df=df.merge(subframe[["Country", "Date", "add"]], how="left", left_on=["Country", "Date"], right_on=["Country", "Date"])

    return work 
train=cumulative_to_adds(train, "ConfirmedCases", "ConfirmedAdds")

train[train.Country=="Algeria"].sort_values("ConfirmedCases", ascending=False)
train=cumulative_to_adds(train, "Fatalities", "FatalitiesAdds")

train=train.drop(["ConfirmedCases", "Fatalities"], axis=1)

train=train.rename(columns={"ConfirmedAdds": "ConfirmedCases", "FatalitiesAdds":"Fatalities"})
#X_test=cumulative_to_adds(X_test, "ConfirmedCases", "ConfirmedAdds")

#X_test=cumulative_to_adds(X_test, "Fatalities", "FatalitiesAdds")

#X_test.drop(["ConfirmedCases", "Fatalities"], axis=1)

#X_test.rename(columns={"ConfirmedAdds": "ConfirmedCases", "FatalitiesAdds":"Fatalities"})
from sklearn.preprocessing import LabelEncoder 



labelencoder = LabelEncoder()

province=labelencoder.fit_transform(train["Province"])

train=pd.concat([train, pd.DataFrame(province)], axis=1)


train=train.drop(["Country","MinDate", "Province"], axis=1)

#train=train.drop(["Country","MinDate"], axis=1)
train.plot(x="Date", y="ConfirmedCases")
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
X_train.info()
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
X_train.head()
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



test["DaysFrom1stCase"] =X_test.DaysFrom1stCase

test["DaysFromBeginning"] = X_test[0]
train.head()
X_test.head()
#X_test= pca.transform(X_test)



#y_pred=regressor.predict(X_test)

y_pred_CC = gbm_CC.predict(X_test, num_iteration=gbm_CC.best_iteration)

y_pred_F = gbm_F.predict(X_test, num_iteration=gbm_F.best_iteration)

forecastId=test.ForecastId.to_numpy()

submission_CC=pd.DataFrame(y_pred_CC)

submission_CC=submission_CC.rename(columns={0:"ConfirmedAdd"})

submission_F=pd.DataFrame(y_pred_F)

submission_F=submission_F.rename(columns={0:"FatalitiesAdd"})

forecastIdDF=pd.DataFrame(forecastId)

forecastIdDF=forecastIdDF.rename(columns={0:"ForecastId"})

submission=pd.concat([forecastIdDF, submission_CC, submission_F ], axis=1)

submission=submission.drop("ForecastId", axis=1)
test=pd.concat([test, submission ], axis=1)
test.head()
test.groupby("Date").ConfirmedAdd.sum().plot()
test=adds_to_cumulative(test, "ConfirmedAdd", "ConfirmedCases")

test=adds_to_cumulative(test, "FatalitiesAdd", "Fatalities")
test[test.Country=='Algeria'].sort_values("Date").head(120)
test.head()
submission.info()
test.head()
test.groupby("DaysFrom1stCase").ConfirmedCases.sum().plot()
test.plot(x="Date", y="ConfirmedCases")
submission=pd.DataFrame(test[["ForecastId","ConfirmedCases", "Fatalities"]])
submission.reset_index()

submission.head()
submission.to_csv("submission.csv", index=False)