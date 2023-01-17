#import libs

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder







#TODO: add lag features with https://medium.com/@NatalieOlivo/use-pandas-to-lag-your-timeseries-data-in-order-to-examine-causal-relationships-f8186451b3a9



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#train = pd.read_csv("train.csv")

#test = pd.read_csv("test.csv")

#region_metadata = pd.read_csv("region_metadata.csv")

#region_date_metadata = pd.read_csv("region_date_metadata.csv")

# Load Data

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
#pandas profiling over data to check for NaNs etc

import pandas_profiling as pp



pp.ProfileReport(train)
#fix data etc

def fixData(input_set):

    input_set.rename(columns={'Country_Region':'Country'}, inplace=True) #Rename columns

    input_set.rename(columns={'Province_State':'State'}, inplace=True)   #Rename columns

    input_set['Date'] = pd.to_datetime(input_set['Date'], infer_datetime_format=True) # change date

    input_set['Date'] = input_set.Date.dt.strftime("%m%d") # convert format to month-day 

    input_set['Date']  = input_set['Date'].astype(int) # convert to int

    input_set["State"].fillna("",inplace=True) # fill with ""

    input_set["CountryState"] = input_set["Country"] + input_set["State"]

    return input_set



#train2 = fixData(train)

train = fixData(train)

test = fixData(test)
train.head()  
test.head()
#build model and encode

from warnings import filterwarnings

filterwarnings('ignore')

from xgboost import XGBRegressor as boostmodel

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score, log_loss

import math



label = LabelEncoder()



sub = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

CountryState = train.CountryState.unique()

for CS in CountryState:

    train2 = train[train["CountryState"] == CS]

    test2 = test[test["CountryState"] == CS]

    train2.CountryState = label.fit_transform(train2.CountryState)

    X = train2[['CountryState', 'Date']]

    Y = train2[['ConfirmedCases']]

    eval_set  = [(X,Y)]

    model1 = boostmodel(learning_rate=0.3,silent=0, n_estimators=1000)

    model1.fit(X, Y,eval_set=eval_set,early_stopping_rounds=100)

    testX = test2[['CountryState', 'Date']]

    testX.CountryState = label.fit_transform(testX.CountryState)

    ConfirmedCases_Pred = model1.predict(testX)

    X = train2[['CountryState', 'Date']]

    Y = train2[['Fatalities']]

    eval_set  = [(X,Y)]

    model2 = boostmodel(learning_rate=0.3,silent=0, n_estimators=1020)

    model2.fit(X, Y,eval_set=eval_set,early_stopping_rounds=100)

    testX = test2[['CountryState', 'Date']]

    testX.CountryState = label.fit_transform(testX.CountryState)

    Fatalities_Pred = model2.predict(testX)

    XForecastId = test2.loc[:, 'ForecastId']

    output = pd.DataFrame({'ForecastId': XForecastId, 'ConfirmedCases': ConfirmedCases_Pred, 'Fatalities': Fatalities_Pred})

    sub = pd.concat([sub, output], axis=0)

round(sub,1).head()

#finaloutput.head(20)
sub.ConfirmedCases.apply(math.floor)

sub.ForecastId = sub.ForecastId.astype('int')

sub.ConfirmedCases = round(sub.ConfirmedCases,1)

sub.Fatalities = round(sub.Fatalities,1)

sub = sub[['ForecastId','ConfirmedCases','Fatalities']]
#creating final submission file

sub.to_csv("submission.csv",index=False) 