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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from datetime import timedelta 

%matplotlib inline
path = "/kaggle/input/covid19-global-forecasting-week-2/"
train_data = pd.read_csv(path+"train.csv")

train_df = train_data

train_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(train_data['Country_Region'], train_data['Province_State'])]

train_df['Date'] = pd.to_datetime(train_df['Date'])

full_data = train_df
today = full_data['Date'].max()+timedelta(days=1) 

#today = '2020-03-12'
# remove date leakage

#train_df = train_df[train_df['Date']<pd.to_datetime(today)]

#train_df.head()
def get_country_data(train_df, area, metric):

    country_data = train_df[train_df['area']==area]

    country_data = country_data.drop(['Id','Province_State', 'Country_Region'], axis=1)

    country_data = pd.pivot_table(country_data, values=['ConfirmedCases','Fatalities'], index=['Date'], aggfunc=np.sum) 

    country_data = country_data[country_data[metric]!=0]

    return country_data        
area_info = pd.DataFrame(columns=['area', 'cases_start_date', 'deaths_start_date', 'init_ConfirmedCases', 'init_Fatalities'])

for i in range(len(train_df['area'].unique())):

    area = train_df['area'].unique()[i]

    area_cases_data = get_country_data(train_df, area, 'ConfirmedCases')

    area_deaths_data = get_country_data(train_df, area, 'Fatalities')

    cases_start_date = area_cases_data.index.min()

    deaths_start_date = area_deaths_data.index.min()

    if len(area_cases_data) > 0:

        confirmed_cases = max(area_cases_data['ConfirmedCases'])

    else:

        confirmed_cases = 0

    if len(area_deaths_data) > 0:

        fatalities = max(area_deaths_data['Fatalities'])

    else:

        fatalities = 0

    area_info.loc[i] = [area, cases_start_date, deaths_start_date, confirmed_cases, fatalities]

area_info = area_info.fillna(pd.to_datetime(today))

area_info['init_cases_day_no'] = pd.to_datetime(today)-area_info['cases_start_date']

area_info['init_cases_day_no'] = area_info['init_cases_day_no'].dt.days.fillna(0).astype(int)

area_info['init_deaths_day_no'] = pd.to_datetime(today)-area_info['deaths_start_date']

area_info['init_deaths_day_no'] = area_info['init_deaths_day_no'].dt.days.fillna(0).astype(int)

area_info.head()
def log_curve(x, k, x_0, ymax):

    return ymax / (1 + np.exp(-k*(x-x_0)))

    

def log_fit(train_df, area, metric):

    area_data = get_country_data(train_df, area, metric)

    x_data = range(len(area_data.index))

    y_data = area_data[metric]

    if len(y_data) < 5:

        estimated_k = -1  

        estimated_x_0 = -1 

        ymax = -1

    elif max(y_data) == 0:

        estimated_k = -1  

        estimated_x_0 = -1 

        ymax = -1

    else:

        try:

            popt, pcov = curve_fit(log_curve, x_data, y_data, bounds=([0,0,0],np.inf), p0=[0.3,100,10000], maxfev=1000000)

            estimated_k, estimated_x_0, ymax = popt

        except RuntimeError:

            print(area)

            print("Error - curve_fit failed") 

            estimated_k = -1  

            estimated_x_0 = -1 

            ymax = -1

    estimated_parameters = pd.DataFrame(np.array([[area, estimated_k, estimated_x_0, ymax]]), columns=['area', 'k', 'x_0', 'ymax'])

    return estimated_parameters
def get_parameters(metric):

    parameters = pd.DataFrame(columns=['area', 'k', 'x_0', 'ymax'], dtype=np.float)

    for area in train_df['area'].unique():

        estimated_parameters = log_fit(train_df, area, metric)

        parameters = parameters.append(estimated_parameters)

    parameters['k'] = pd.to_numeric(parameters['k'], downcast="float")

    parameters['x_0'] = pd.to_numeric(parameters['x_0'], downcast="float")

    parameters['ymax'] = pd.to_numeric(parameters['ymax'], downcast="float")

    parameters = parameters.replace({'k': {-1: parameters[parameters['ymax']>0].median()[0]}, 

                                     'x_0': {-1: parameters[parameters['ymax']>0].median()[1]}, 

                                     'ymax': {-1: parameters[parameters['ymax']>0].median()[2]}})

    return parameters
cases_parameters = get_parameters('ConfirmedCases')

cases_parameters.head(20)
deaths_parameters = get_parameters('Fatalities')

deaths_parameters.head(20)
fit_df = area_info.merge(cases_parameters, on='area', how='left')

fit_df = fit_df.rename(columns={"k": "cases_k", "x_0": "cases_x_0", "ymax": "cases_ymax"})

fit_df = fit_df.merge(deaths_parameters, on='area', how='left')

fit_df = fit_df.rename(columns={"k": "deaths_k", "x_0": "deaths_x_0", "ymax": "deaths_ymax"})

fit_df['init_ConfirmedCases_fit'] = log_curve(fit_df['init_cases_day_no'], fit_df['cases_k'], fit_df['cases_x_0'], fit_df['cases_ymax'])

fit_df['init_Fatalities_fit'] = log_curve(fit_df['init_deaths_day_no'], fit_df['deaths_k'], fit_df['deaths_x_0'], fit_df['deaths_ymax'])

fit_df['ConfirmedCases_error'] = fit_df['init_ConfirmedCases']-fit_df['init_ConfirmedCases_fit']

fit_df['Fatalities_error'] = fit_df['init_Fatalities']-fit_df['init_Fatalities_fit']

fit_df.head()
test_data = pd.read_csv(path+"test.csv")

test_df = test_data

test_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(test_data['Country_Region'], test_data['Province_State'])]



test_df = test_df.merge(fit_df, on='area', how='left')

#test_df = test_df.merge(cases_parameters, on='area', how='left')

#test_df = test_df.rename(columns={"k": "cases_k", "x_0": "cases_x_0", "ymax": "cases_ymax"})

#test_df = test_df.merge(deaths_parameters, on='area', how='left')

#test_df = test_df.rename(columns={"k": "deaths_k", "x_0": "deaths_x_0", "ymax": "deaths_ymax"})



test_df['Date'] = pd.to_datetime(test_df['Date'])

test_df['cases_start_date'] = pd.to_datetime(test_df['cases_start_date'])

test_df['deaths_start_date'] = pd.to_datetime(test_df['deaths_start_date'])



test_df['cases_day_no'] = test_df['Date']-test_df['cases_start_date']

test_df['cases_day_no'] = test_df['cases_day_no'].dt.days.fillna(0).astype(int)

test_df['deaths_day_no'] = test_df['Date']-test_df['deaths_start_date']

test_df['deaths_day_no'] = test_df['deaths_day_no'].dt.days.fillna(0).astype(int)



test_df['ConfirmedCases_fit'] = log_curve(test_df['cases_day_no'], test_df['cases_k'], test_df['cases_x_0'], test_df['cases_ymax'])

test_df['Fatalities_fit'] = log_curve(test_df['deaths_day_no'], test_df['deaths_k'], test_df['deaths_x_0'], test_df['deaths_ymax'])



test_df['ConfirmedCases_pred'] = round(test_df['ConfirmedCases_fit']+test_df['ConfirmedCases_error'])

test_df['Fatalities_pred'] = round(test_df['Fatalities_fit']+test_df['Fatalities_error'])



test_df.head()
# generate submission

submission = pd.DataFrame(data={'ForecastId': test_df['ForecastId'], 'ConfirmedCases': test_df['ConfirmedCases_pred'], 'Fatalities': test_df['Fatalities_pred']}).fillna(0.5)

submission.to_csv("/kaggle/working/submission.csv", index=False)
submission.head()