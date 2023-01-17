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


from pandas_profiling import ProfileReport
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
train_profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})

train_profile
train.rename(columns={'Country_Region':'country'}, inplace=True)

test.rename(columns={'Country_Region':'country'}, inplace=True)



train.rename(columns={'Province_State':'state'}, inplace=True)

test.rename(columns={'Province_State':'state'}, inplace=True)



train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)



train.info()

test.info()



y1Train = train.iloc[:, -2]

y1Train.head()

y2Train = train.iloc[:, -1]

y2Train.head()



EMPTY_VAL = "EMPTY_VAL"



#fill the empty states with EMPTY_VAL

def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
XTrain = train.copy()



XTrain['state'].fillna(EMPTY_VAL, inplace=True)

XTrain['state'] = XTrain.loc[:, ['state', 'country']].apply(lambda x : fillState(x['state'], x['country']), axis=1)



XTrain.loc[:, 'Date'] = XTrain.Date.dt.strftime("%m%d")

XTrain["Date"]  = XTrain["Date"].astype(int)



XTrain.head()



#X_Test = df_test.loc[:, ['State', 'Country', 'Date']]

XTest = test.copy()



XTest['state'].fillna(EMPTY_VAL, inplace=True)

XTest['state'] = XTest.loc[:, ['state', 'country']].apply(lambda x : fillState(x['state'], x['country']), axis=1)



XTest.loc[:, 'Date'] = XTest.Date.dt.strftime("%m%d")

XTest["Date"]  = XTest["Date"].astype(int)



XTest.head()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()



XTrain.country = le.fit_transform(XTrain.country)

XTrain['state'] = le.fit_transform(XTrain['state'])



XTrain.head()



XTest.country = le.fit_transform(XTest.country)

XTest['state'] = le.fit_transform(XTest['state'])



XTest.head()



train.head()

train.loc[train.country == 'Afghanistan', :]

test.tail()
from warnings import filterwarnings

filterwarnings('ignore')



from sklearn import preprocessing



le = preprocessing.LabelEncoder()





from sklearn.ensemble import RandomForestClassifier



countries = XTrain.country.unique()
# Predict data and Create submission file from test data

out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = XTrain.loc[XTrain.country == country, :].state.unique()

    

    # check whether string is nan or not

    for state in states:

        XTrain_CS = XTrain.loc[(XTrain.country == country) & (XTrain.state == state), ['state', 'country', 'Date', 'ConfirmedCases', 'Fatalities']]

        

        y1Train_CS = XTrain_CS.loc[:, 'ConfirmedCases']

        y2Train_CS = XTrain_CS.loc[:, 'Fatalities']

        

        XTrain_CS = XTrain_CS.loc[:, ['state', 'country', 'Date']]

        

        XTrain_CS.Country = le.fit_transform(XTrain_CS.country)

        XTrain_CS['state'] = le.fit_transform(XTrain_CS['state'])

        

        XTest_CS = XTest.loc[(XTest.country == country) & (XTest.state == state), ['state', 'country', 'Date', 'ForecastId']]

        

        XTest_CS_Id = XTest_CS.loc[:, 'ForecastId']

        XTest_CS = XTest_CS.loc[:, ['state', 'country', 'Date']]

        

        XTest_CS.country = le.fit_transform(XTest_CS.country)

        XTest_CS['state'] = le.fit_transform(XTest_CS['state'])



        

        model1 = RandomForestClassifier(n_estimators=1000)

        model1.fit(XTrain_CS, y1Train_CS)

        y1pred = model1.predict(XTest_CS)

        

        model2 = RandomForestClassifier(n_estimators=1000)

        model2.fit(XTrain_CS, y2Train_CS)

        y2pred = model2.predict(XTest_CS)

        

        data = pd.DataFrame({'ForecastId': XTest_CS_Id, 'ConfirmedCases': y1pred, 'Fatalities': y2pred})

        out = pd.concat([out, data], axis=0)

    

    
out.ForecastId = out.ForecastId.astype('int')

out.tail()

out.to_csv('submission.csv', index=False)

print("Submission file Created.....")