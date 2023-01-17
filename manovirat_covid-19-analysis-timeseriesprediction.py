# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

import csv

import random

from fbprophet import Prophet



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-4/train.csv',parse_dates = ['Date'])

test = pd.read_csv ('/kaggle/input/covid19-global-forecasting-week-4/test.csv',parse_dates = ['Date'])

train['Country_State']=train.Country_Region+train.Province_State.fillna('')

test['Country_State']=test.Country_Region+test.Province_State.fillna('')

train.head()
train_df = train[['Date','ConfirmedCases','Fatalities','Country_State']]

train_df.head()
train_df.tail()
test_df = test[['ForecastId','Country_State']]
countriesorstates = train_df["Country_State"].unique()

print(len(countriesorstates))
sub = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

strt = '2020-04-02'

end = '2020-05-14'

for value in countriesorstates:

    random.seed(0)

    #train_model_cc = Prophet(seasonality_mode = 'additive',interval_width=0.95)

    #train_model_ft = Prophet(seasonality_mode = 'additive',interval_width=0.95)

    

    train_model_cc = Prophet(seasonality_mode = 'multiplicative',changepoint_prior_scale = 30, interval_width=0.95)

    train_model_ft = Prophet(seasonality_mode = 'multiplicative',changepoint_prior_scale = 30, interval_width=0.95)

    #Data Filter

    train_temp = train_df.loc[train_df["Country_State"] == value]

    test_temp = test_df.loc[test_df["Country_State"] == value]

    ###Confirmed Cases

    train_temp_cc = train_temp[['Date','ConfirmedCases']]

    train_temp_cc = train_temp_cc.sort_values(by = ['Date'])

    train_temp_cc = train_temp_cc.rename(columns = {'Date':'ds','ConfirmedCases':'y'})

    #train_temp_cc['y']=np.log1p(train_temp_cc['y'])



    #Fatalities Cases

    train_temp_ft = train_temp[['Date','Fatalities']]

    train_temp_ft = train_temp_ft.sort_values(by = ['Date'])

    train_temp_ft = train_temp_ft.rename(columns = {'Date':'ds','Fatalities':'y'})

    #train_temp_ft['y']=np.log1p(train_temp_ft['y'])

    

    ###Confirmed Modeling

    train_model_cc.fit(train_temp_cc)

    train_forecast_cc = train_model_cc.make_future_dataframe(periods=43, freq='D',include_history = True)

    train_forecast_cc = train_model_cc.predict(train_forecast_cc)

    reg_cc = train_forecast_cc[['ds','yhat']]

    test_cc = reg_cc[(reg_cc['ds']>=strt) & (reg_cc['ds']<=end)]

    test_cc = test_cc.yhat.values

    

    ###Fatalities Modeling

    train_model_ft.fit(train_temp_ft)

    train_forecast_ft = train_model_ft.make_future_dataframe(periods=43, freq='D',include_history = True)

    train_forecast_ft = train_model_ft.predict(train_forecast_ft)

    reg_ft = train_forecast_ft[['ds','yhat']]

    test_ft = reg_ft[(reg_ft['ds']>=strt) & (reg_ft['ds']<=end)]

    test_ft = test_ft.yhat.values

    

    test_cc = test_cc.flatten()

    test_ft = test_ft.flatten()

    

    sub_temp = pd.DataFrame({'ForecastId': test_temp["ForecastId"].loc[test_temp["Country_State"] == value],

                             'ConfirmedCases': test_cc, 'Fatalities': test_ft})

    #sub_temp = pd.DataFrame({'ForecastId': test_temp["ForecastId"].loc[test_temp["Country_State"] == value],

    #                         'ConfirmedCases': np.exp(test_cc), 'Fatalities': np.exp(test_ft)})

    sub = pd.concat([sub, sub_temp], axis = 0)
sub.ForecastId = sub.ForecastId.astype('int')

for i in range(len(sub)):

    sub["ConfirmedCases"][i] = int(round(sub["ConfirmedCases"][i]))

    sub["Fatalities"][i] = int(round(sub["Fatalities"][i]))



sub.to_csv("submission.csv", index = False)