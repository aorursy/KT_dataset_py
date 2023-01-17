!pip install pmdarima

from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace import sarimax
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
import pandas as pd

import numpy as np
total = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
total.head()
test.head()
unique_countries = np.unique(total['Country_Region'])

unique_countries


total['Province_State'] = total['Province_State'].fillna('None')
u_regions = {}



for country in unique_countries:

    df = total[total['Country_Region'] == country]

    region = df['Province_State'].values[0]

    if region is not 'None':

        regions = np.unique(df['Province_State'])

        u_regions[country] = regions

print(u_regions)


total_specific_df = total[total['Country_Region'] == unique_countries[0]]



test_specific_df = test[test['Country_Region'] == unique_countries[0]]

test_length = len(test_specific_df['Date'])

print(test_specific_df['Date'][0])

test_date = test_specific_df['Date'][0]

# train = total

train = total[total['Date'] < test_date]



train_specific_df = train[train['Country_Region'] == unique_countries[0]]



total_length = len(total_specific_df['Date'])

common = len(total_specific_df['Date']) - len(train_specific_df['Date'])

print(common, total_length)

train = total

# print(train[train['Country/Region'] == unique_countries[0]])
def run_model(df, value):

#     print(df)

#     print(df['Country_Region'].values[0])

#     print(value)

    try:

        model = pm.auto_arima(df[value], start_p=1, start_q=1,

                              test='adf',       

                              max_p=7, max_q=7,

                              m=1,            

                              d=None,         

                              seasonal=True,

                              start_P=0, 

                              D=0, 

                              trace=False,

                              error_action='ignore',  

                              suppress_warnings=True, 

                              stepwise=True)



    #     print(model.summary())

    #     print(model.order)



    #     model= sarimax.SARIMAX(df[value],

    #          order=model.order, enforce_stationarity=True)

#         model_fit = model.fit(df[value], disp=0)

    #     print(model_fit.summary())



        forecast_1= model.predict(test_length-common)

    except np.linalg.LinAlgError as E:

        print(E)

        forecast_1 = np.zeros(test_length-common)

    return forecast_1


result_ConfirmedCases = {}





for country in unique_countries:

    print(country)

    if u_regions.get(country) is None:

        df = train[train['Country_Region'] == country]

        forecast_1 = run_model(df, 'ConfirmedCases')

        result_ConfirmedCases[country] = np.concatenate((df['ConfirmedCases'][-1*common:] , forecast_1), axis=0)    

    else:

        for region in u_regions[country]:

            df1 = train[train['Country_Region'] == country]

            df = df1[df1['Province_State'] == region]

            forecast_1 = run_model(df, 'ConfirmedCases')

            if country not in result_ConfirmedCases:

                result_ConfirmedCases[country] = np.concatenate((df['ConfirmedCases'][-1*common:], forecast_1), axis=0)

            else:

                result_ConfirmedCases[country] = np.concatenate((result_ConfirmedCases[country], np.concatenate((df['ConfirmedCases'][-1*common:], forecast_1), axis=0)), axis=0)

    
result_Fatalities = {}

for country in unique_countries:

    print(country)

    if u_regions.get(country) is None:

        df = train[train['Country_Region'] == country]

        forecast_1 = run_model(df, 'Fatalities')

        result_Fatalities[country] = np.concatenate((df['Fatalities'][-1*common:] , forecast_1), axis=0)

    else:

        for region in u_regions[country]:

            df1 = train[train['Country_Region'] == country]

            df = df1[df1['Province_State'] == region]

            forecast_1 = run_model(df, 'Fatalities')

            if country not in result_Fatalities:

                result_Fatalities[country] = np.concatenate((df['Fatalities'][-1*common:], forecast_1), axis=0)

            else:

                result_Fatalities[country] = np.concatenate((result_Fatalities[country], np.concatenate((df['Fatalities'][-1*common:], forecast_1), axis=0)), axis=0)
total_fat = 0

total_case = 0



for coutry in result_Fatalities:

    

    total_fat += result_Fatalities[coutry].shape[0]

#     print(result_Fatalities[country])

print(total_fat)



for country in result_ConfirmedCases:

    total_case += result_ConfirmedCases[country].shape[0]

print(total_case)





prediction_1 = []

prediction_2 = []

forecast_id = [_ for _ in range(1, total_fat+1)]

for country in unique_countries:

    prediction_1 += result_ConfirmedCases[country].tolist()

    prediction_2 += result_Fatalities[country].tolist()

    

print(len(prediction_1))

print(len(prediction_2))

print(len(forecast_id))



# prediction_1 = df['Predicted_ConfirmedCases']

# prediction_2 = df['Predicted_Fatalities']



# # Submit predictions

prediction_1 = [int(item) for item in list(map(round, prediction_1))]

prediction_2 = [int(item) for item in list(map(round, prediction_2))]





submission = pd.DataFrame({

    "ForecastId": forecast_id, 

    "ConfirmedCases": prediction_1, 

    "Fatalities": prediction_2

})



print(submission)

submission.to_csv('submission.csv', index=False)
