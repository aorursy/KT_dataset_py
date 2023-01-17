# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





#submission = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv", parse_dates=["Date"])

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv", parse_dates=["Date"])

global_data = pd.read_csv("../input/external/Global_Data_by_Country_2019.csv")

coordinates=pd.read_csv("../input/external/coordinates.csv")

restrictions =pd.read_csv("../input/restrictions/mainrestrictions.csv", sep=";", parse_dates=["BorderClosure","Curfews","DomesticTravel","FullLockdown","InternationalFlights","MassTesting","PartialLockdown","ServicesClosure","SchoolsClosure"])



restrictions.rename(columns={"Country": "RestCountry"}, inplace=True)

type(restrictions.BorderClosure[0])

#type(train.Date[0])
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

        temparray[-1]=temparray[-1]+temparray[-2]

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



train=train.merge(restrictions, how='left', left_on=["Country"], right_on="RestCountry")

for col in restrictions.drop("RestCountry", axis=1).columns:

    train[col + "days"] = (train[col]-train["MinDate"]).dt.days

    train[col + "days"]=train[col + "days"].fillna(0)



to_drop=restrictions.columns

train=train.drop(to_drop,axis=1)    





train.describe()
train[train.Country=="Algeria"].sort_values("Date", ascending=False)
train=cumulative_to_adds(train, "ConfirmedCases", "ConfirmedCumul")

train=train.drop("ConfirmedCases", axis=1)

train=train.rename(columns={"ConfirmedCumul":"ConfirmedCases"})

train.head()

train[train.Country=="Algeria"].sort_values("Date", ascending=False)
train.head()
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

countrylabelencoder = LabelEncoder()

province=labelencoder.fit_transform(train["Province"])



country=countrylabelencoder.fit_transform(train["Country"])



train=pd.concat([train, pd.DataFrame(province)], axis=1)

train=train.rename(columns={0: "Province_Encoded"})

train=pd.concat([train,  pd.DataFrame(country)], axis=1)

train=train.rename(columns={0: "Country_Encoded"})
train.head()


train=train.drop(["MinDate", "Province", "Country"], axis=1)

#train=train.drop(["Country","MinDate"], axis=1)
train[train.ConfirmedCases>0].head(100)
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor





#X_train.head()
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb





### Columns to drop after selection





threshold=train.DaysFrom1stCase.max()*.7

minrmse_CC=train["ConfirmedCases"].max()**2

minrmse_F=train["Fatalities"].max()**2





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



collist=train.drop(["ConfirmedCases", "Fatalities",  "Date",  "DaysFrom1stCase"], axis=1)

list_CC= ['Id']

#'HealthExpenditure', 'Province_Encoded', 'BorderClosuredays', 'Curfewsdays' , 'InternationalFlightsdays', 'MassTestingdays',

#         'ServicesClosuredays', 'SchoolsClosuredays']

collist=collist.drop(list_CC, axis=1)





model_CC = RandomForestRegressor(n_estimators = 10, random_state = 0)



model_F = RandomForestRegressor(n_estimators = 10, random_state = 0)







from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.impute import SimpleImputer 





f= SimpleImputer(missing_values= np.nan, strategy='median')







y_train_CC=train.loc[:,"ConfirmedCases"].fillna(train.loc[:,"ConfirmedCases"].median())

y_train_F=train.loc[:, "Fatalities"].fillna(train.loc[:,"Fatalities"].median())

X_train=train.drop(["ConfirmedCases", "Fatalities",  "Date", "Id"], axis=1)

    

f.fit(X_train)

X_train=f.transform(X_train)

    

    



y_train_CC[y_train_CC.isna()]


model_CC.fit(X_train, y_train_CC)

model_F.fit(X_train, y_train_F)


test.head()


## add days since first case column ##

X_test=X_test.merge(mindatesDF, how='left', left_on="Province", right_on="Province")

X_test["DaysFrom1stCase"]=(X_test["Date"]-X_test["MinDate"]).dt.days

X_test.loc [X_test["DaysFrom1stCase"]<0 , "DaysFrom1stCase"] =0

X_test=X_test.merge(restrictions, how='left', left_on=["Country"], right_on="RestCountry")

for col in restrictions.drop("RestCountry", axis=1).columns:

    X_test[col + "days"] = (X_test[col]-X_test["MinDate"]).dt.days

    X_test[col + "days"]=X_test[col + "days"].fillna(0)



to_drop=restrictions.columns

X_test=X_test.drop(to_drop,axis=1)    



X_test=X_test.drop(["MinDate","ForecastId", "Date"], axis=1)
province=labelencoder.transform(X_test["Province"])



country=countrylabelencoder.transform(X_test["Country"])



X_test=pd.concat([X_test,pd.DataFrame(province) ], axis=1)

X_test=X_test.rename(columns={0:"Province_Encoded"})

X_test=pd.concat([X_test, pd.DataFrame(country) ], axis=1)

X_test=X_test.rename(columns={0:"Country_Encoded"})

X_test=X_test.drop(["Province", "Country"], axis=1)



for col in X_test.columns:

    X_test[col]=X_test[col].fillna(X_test[col].median())
#X_test= pca.transform(X_test)



#y_pred=regressor.predict(X_test)

y_pred_CC = model_CC.predict(X_test)

y_pred_F = model_F.predict(X_test)

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
submission.head()
test=pd.concat([test, submission ], axis=1)





test[test["Country"]=="France"]
test=adds_to_cumulative(test, "ConfirmedCases", "ConfirmedAdds")

test=test.drop("ConfirmedCases", axis=1)

test=test.rename(columns={"ConfirmedAdds":"ConfirmedCases"})

submission.to_csv("submission.csv", index=False)
test[test["Country"]=="Algeria"]
X_test.head()

test=pd.concat([test,X_test], axis=1 )
import seaborn as sns

sns.set(style="darkgrid")



# Load an example dataset with long-form data





# Plot the responses for different events and regions

sns.lineplot(x="DaysFrom1stCase", y="ConfirmedCases",

             hue="Country", 

             data=test[test["Country"] =="China"])