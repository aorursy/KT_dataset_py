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

import scipy.optimize as optim

import matplotlib.pyplot as plt

from scipy import stats

import csv

from datetime import datetime

train_data = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

train_data = train_data.replace(np.nan, '', regex=True)

test_data = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

test_data = test_data.replace(np.nan, '', regex=True)
# Timestep counts how many days have passed since the observations started taking place



def filter_train_data(country, region):

    selector = train_data['Country_Region'] == country

    onlyonecountry = train_data[selector]

    selector2 = onlyonecountry['Province_State'] == region

    onlyoneregion = onlyonecountry[selector2]

    onlyoneregion = onlyoneregion.reset_index(drop=False)

    del onlyoneregion['index']

    del onlyoneregion['Id']

    onlyoneregion['Timestep'] = onlyoneregion.index

    return onlyoneregion
def infection_start(country, region):

    try:

        infection_start_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

        infection_start_df = infection_start_df.replace(np.nan, '', regex=True)

        selector = infection_start_df['Country_Region'] == country

        onlyonecountry = infection_start_df[selector]

        selector2 = onlyonecountry['Province_State'] == region

        onlyoneregion=onlyonecountry[selector2]

        return onlyoneregion['ConfirmedCases'].iloc[0]

    except:

        return 2

    

infection_start('Brazil','')
#what is the population of a certain country or region?



population_reader = csv.reader(open('../input/population/population.csv', 'r'))

population_dict = {}

next(population_reader)

for row in population_reader:

   k, v = row

   population_dict[k] = int(v)
pop_by_region_reader = csv.reader(open('../input/populationbycity/populationbycity.csv', 'r'))

populationbyregion_dict = {}

next(pop_by_region_reader)

for row in pop_by_region_reader:

   k, v = row

   populationbyregion_dict[k] = int(v)
def get_population(country,region):

    if region != '':

        if region in populationbyregion_dict:

            return  populationbyregion_dict[region]

        else:

            return 1000000

    elif country  != '':

        if country in population_dict:

            return population_dict[country]

        else:

            return 1000000

    else:

        return 1000000

print(get_population('Denmark',''))
#71 days passed between the first observation and the first day to be forecasted

# here we also filter by date, so we are able to forecast the result for any particular day



def filter_test_data(country, region,date):

    selector = test_data['Country_Region'] == country

    onlyonecountry = test_data[selector]

    selector2 = onlyonecountry['Province_State'] == region

    onlyoneregion = onlyonecountry[selector2]

    onlyoneregion = onlyoneregion.reset_index(drop=False)

    del onlyoneregion['index']

    onlyoneregion['Timestep'] = onlyoneregion.index + 71

    if (date != ''):

        dateselect = test_data['Date'] == date

        onlyoneregion = onlyoneregion[dateselect]

    return onlyoneregion

panama = filter_train_data ('Panama','')

panama.head()
panama_test = filter_test_data ('Panama','','2020-05-11')

panama_test.head()
def my_logistic(t, a, b, c):

    return c / (1 + a * np.exp(-b*t))
def calculate_infection(country_entry, region_entry,date_entry):

    train_set = filter_train_data (country_entry,region_entry)

    local_x = np.array(train_set['Timestep']) + 1

    local_y = np.array(train_set['ConfirmedCases'])

    p0 = np.random.exponential(size=3)

    bounds = (0, [100000., 3., 1000000000.])

    try:

        (a,b,c),cov = optim.curve_fit(my_logistic, local_x, local_y,p0=p0,bounds=bounds)

    except:

        initial_infected = infection_start(country_entry, region_entry)

        c=get_population(country_entry,region_entry)

        b=2

        a=c-1

    test_set=filter_test_data(country_entry,region_entry,date_entry)

    test_set['Infected'] = round(my_logistic(test_set['Timestep'],a,b,c))

    return test_set.iloc[0]['Infected']

    

    
uruguay_forecast = calculate_infection('Uruguay','','2020-05-11')

print(uruguay_forecast)
def calculate_fatalities(country_entry, region_entry,date):

    df = filter_train_data (country_entry,region_entry)

    X = df['Timestep'].values

    Y = df['Fatalities'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)

    forecast_day = filter_test_data(country_entry,region_entry,date).iloc[0]['Timestep']

    return round(slope*forecast_day+intercept)








submission_file = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

test_data_to_forecast  = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

test_data_to_forecast = test_data_to_forecast.replace(np.nan, '', regex=True)

merged_inner = pd.merge(submission_file, test_data_to_forecast, on='ForecastId')





beginning = 0

end = 13458



with open('submission.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["ForecastId", "ConfirmedCases", "Fatalities"])

    for j in range(beginning, end + 1):

        try:

            forecast_Id =  merged_inner.iloc[j,0]

            infected = calculate_infection(merged_inner.iloc[j,4],merged_inner.iloc[j,3],merged_inner.iloc[j,5])

            casualties =  calculate_fatalities(merged_inner.iloc[j,4],merged_inner.iloc[j,3],merged_inner.iloc[j,5])

            lst = [int(forecast_Id),int(infected) , int(casualties)]

            print(*lst, sep = ", ")

            writer.writerow(lst)

        except:

            print('error:' + str(forecast_Id) + ' ')

            lst = [int(forecast_Id),0 , 0]

            print(*lst, sep = ", ")

            writer.writerow(lst)

            continue

    print('End')

     