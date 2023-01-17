# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Any results you write to the current directory are saved as output.
# EDA
# for train data set
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
train.rename(columns={'Country_Region':'Country'}, inplace=True) #Rename columns
train.rename(columns={'Province_State':'State'}, inplace=True)   #Rename columns
train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True) # change date
train['Date'] = train.Date.dt.strftime("%m%d") # convert format to month-day 
train['Date']  = train['Date'].astype(int) # convert to int
train["State"].fillna("",inplace=True) # fill with ""
train["CountryState"] = train["Country"] + train["State"]
print("train done")

# for test data set
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
test.rename(columns={'Country_Region':'Country'}, inplace=True) #Rename columns
test.rename(columns={'Province_State':'State'}, inplace=True)   #Rename columns
test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True) # change date
test['Date'] = test.Date.dt.strftime("%m%d") # convert format to month-day 
test['Date']  = test['Date'].astype(int) # convert to int
test["State"].fillna("",inplace=True) # fill with ""
test["CountryState"] = test["Country"] + test["State"]
print("test done")

test[test["CountryState"] == "India"].head(20)
%%time
from warnings import filterwarnings
filterwarnings('ignore')
from xgboost import XGBRegressor as boostmodel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, log_loss
import math
le = LabelEncoder()
finaloutput = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
CountryState = train.CountryState.unique()
for CS in CountryState:
    #print(CS)
    trainIndia = train[train["CountryState"] == CS]
    #trainIndia =  trainIndia[trainIndia["ConfirmedCases"] > 0 ]
    testIndia = test[test["CountryState"] == CS]
    trainIndia.CountryState = le.fit_transform(trainIndia.CountryState)
    X = trainIndia[['CountryState', 'Date']]
    Y = trainIndia[['ConfirmedCases']]
    # confifrmed cases
    eval_set  = [(X,Y)]
    model1 = boostmodel(learning_rate=0.3,silent=0, n_estimators=1000)
    model1.fit(X, Y,eval_set=eval_set,early_stopping_rounds=100)
    #model1.fit(X, Y)
    testX = testIndia[['CountryState', 'Date']]
    testX.CountryState = le.fit_transform(testX.CountryState)
    ConfirmedCases_Pred = model1.predict(testX)
    # fatalities
    X = trainIndia[['CountryState', 'Date']]
    Y = trainIndia[['Fatalities']]
    eval_set  = [(X,Y)]
    model2 = boostmodel(learning_rate=0.3,silent=0, n_estimators=1000)
    model2.fit(X, Y,eval_set=eval_set,early_stopping_rounds=100)
    #model2.fit(X, Y,eval_set=eval_set,early_stopping_rounds=100)
    testX = testIndia[['CountryState', 'Date']]
    testX.CountryState = le.fit_transform(testX.CountryState)
    Fatalities_Pred = model2.predict(testX)
    #print(ConfirmedCases_Pred)
    XForecastId = testIndia.loc[:, 'ForecastId']
    output = pd.DataFrame({'ForecastId': XForecastId, 'ConfirmedCases': ConfirmedCases_Pred, 'Fatalities': Fatalities_Pred})
    finaloutput = pd.concat([finaloutput, output], axis=0)
    #print("output")
print("Program completed")   
round(finaloutput,1).head()
#finaloutput.head(20)
finaloutput.ConfirmedCases.apply(math.floor)
finaloutput.ForecastId = finaloutput.ForecastId.astype('int') # convert
#finaloutput.ConfirmedCases = finaloutput.ConfirmedCases.apply(math.ceil)
finaloutput.ConfirmedCases = round(finaloutput.ConfirmedCases,1)
#finaloutput.Fatalities = finaloutput.Fatalities.apply(math.ceil)
finaloutput.Fatalities = round(finaloutput.Fatalities,1)
finaloutput = finaloutput[['ForecastId','ConfirmedCases','Fatalities']]
#creating final submission file
finaloutput.to_csv("submission.csv",index=False) 
print("done")
