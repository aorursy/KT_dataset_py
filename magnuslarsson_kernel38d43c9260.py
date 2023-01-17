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

import seaborn as sns

from datetime import datetime
print('Importing training and test data')

train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')



# Update dataframe

train_df['Province_State'] = train_df['Province_State'].fillna('')

train_df['Region'] = train_df['Country_Region'] + train_df['Province_State']



test_df['Province_State'] = test_df['Province_State'].fillna('')

test_df['Region'] = test_df['Country_Region'] + test_df['Province_State']



regions = train_df.Region.unique()



# Match days in train and test

train_min_date = train_df[train_df['Region']=='Sweden']['Date'].min()

test_min_date = test_df[test_df['Region']=='Sweden']['Date'].min()

dt_train_min = datetime.strptime(train_min_date, '%Y-%m-%d') 

dt_test_min = datetime.strptime(test_min_date, '%Y-%m-%d') 



test_start_day = dt_test_min.timetuple().tm_yday - dt_train_min.timetuple().tm_yday

print(test_start_day)



# Extract dataframes for each country

train_data = {}

test_data = {}

for region in regions:

    train_data[region] = train_df[train_df['Region']==region]    

    train_data[region]['DayNo'] = np.arange(len(train_df[train_df['Region']==region]['Date']))

    test_data[region] = test_df[test_df['Region']==region]    

    test_data[region]['DayNo'] = np.arange(test_start_day,test_start_day+len(test_df[test_df['Region']==region]['Date']))
train_max_date = train_df[train_df['Region']=='Sweden']['Date'].max()

dt_train_max = datetime.strptime(train_max_date, '%Y-%m-%d')

train_max_day = dt_train_max.timetuple().tm_yday - dt_train_min.timetuple().tm_yday

print(train_max_day)

#int(train_data["Sweden"][train_data["Sweden"]['DayNo']==69]['ConfirmedCases'].tolist()[0])

#test_data["Sweden"]['DayNo']
def sigmoid(x, a, b, c):

    return a*np.exp(c*(x-b))/(np.exp(c*(x-b))+1)
failed_confirmed = []

failed_fatalities = []

confirmed_popt = {}

fatalities_popt = {}



for region in regions:

    x_data = train_data[region]['DayNo']

    y_ConfirmedCases_data = train_data[region]['ConfirmedCases']

    y_Fatalities_data = train_data[region]['Fatalities']



    # Fit data to function

    try:

        popt, pcov = curve_fit(sigmoid, x_data, y_ConfirmedCases_data)

        confirmed_popt[region] = popt

    except:

        failed_confirmed.append(region)



    try:

        popt, pcov = curve_fit(sigmoid, x_data, y_Fatalities_data)

        fatalities_popt[region] = popt

    except:

        failed_fatalities.append(region)

        

print("Failed confirmed: " + str(len(failed_confirmed)))

print("Failed fatalities: " + str(len(failed_fatalities)))

print("Total: " + str(len(regions)))
# Handle failed data

confirmed_coeffs = [x for x in confirmed_popt.values()] 

mean_confirmed_coeffs = np.mean(confirmed_coeffs, axis=0)

print(mean_confirmed_coeffs)



fatalities_coeffs = [x for x in fatalities_popt.values()] 

mean_fatalities_coeffs = np.mean(fatalities_coeffs, axis=0)

print(mean_fatalities_coeffs)



for region in failed_confirmed:

    x_data = train_data[region]['DayNo']

    y_ConfirmedCases_data = train_data[region]['ConfirmedCases']

    # Fit data to function

    try:

        popt, pcov = curve_fit(sigmoid, x_data, y_ConfirmedCases_data, maxfev=1000, ftol=1e-5)

        confirmed_popt[region] = popt

    except:

        start = 0

        for data in y_ConfirmedCases_data:

            if data > 0:

                break

            start = start + 1

        popt = mean_confirmed_coeffs

        popt[1] = start

        confirmed_popt[region] = popt

        print("Failed for C " + region + " : " + str(popt))

        

        

for region in failed_fatalities:

    x_data = train_data[region]['DayNo']

    y_Fatalities_data = train_data[region]['Fatalities']



    # Fit data to function

    try:

        popt, pcov = curve_fit(sigmoid, x_data, y_Fatalities_data, maxfev=1000, ftol=1e-5)

        fatalities_popt[region] = popt

    except:

        start = 0

        for data in y_Fatalities_data:

            if data > 0:

                break

            start = start + 1

        popt = mean_fatalities_coeffs

        popt[1] = start

        fatalities_popt[region] = popt

        print("Failed F for " + region + " : " + str(popt))
sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

test_regions = test_df.Region.unique()

total_count = 0



for region in test_regions:

    forecastIds = test_data[region]['ForecastId']

    x_test_data = test_data[region]['DayNo']

    y_conf_test_data = np.nan_to_num(sigmoid(x_test_data, *confirmed_popt[region])).astype(np.int) 

    x_test_data = test_data[region]['DayNo']

    y_fatal_test_data = np.nan_to_num(sigmoid(x_test_data, *fatalities_popt[region])).astype(np.int) 

    idx = 0

    x_test_data = x_test_data.tolist()

    for id in forecastIds:

        day_no = x_test_data[idx]

        row_index = sub.index[sub['ForecastId'] == id]

        if day_no > train_max_day:

            sub.set_value(row_index, 'ConfirmedCases', y_conf_test_data[idx])

            sub.set_value(row_index, 'Fatalities', y_fatal_test_data[idx])

        else:

            sub.set_value(row_index, 'ConfirmedCases', int(train_data[region][train_data[region]['DayNo']==day_no]['ConfirmedCases'].tolist()[0]))

            sub.set_value(row_index, 'Fatalities', int(train_data[region][train_data[region]['DayNo']==day_no]['Fatalities'].tolist()[0])) 

        idx = idx + 1



sub.to_csv('/kaggle/working/submission.csv', index=False) 