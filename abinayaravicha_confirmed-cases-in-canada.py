# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_confirm = pd.read_csv("/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv")

covid_confirm
covid_recov = pd.read_csv("/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv")

covid_recov
covid_death = pd.read_csv("/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv")

covid_death
Country = covid_death['Country/Region']

Country
# extracting data only for Canada 

Canada_data = covid_death.loc[Country=='Canada']

Canada_data
#pre-processing the dataset

Canada_deaths = Canada_data.iloc[:, 4:]

Canada_deaths
#Summation of death each day

Daily_Deaths = Canada_deaths.sum()

Daily_Deaths
# count of death each day

Canada_deaths.loc['Total Deaths'] = Daily_Deaths

Canada_deaths
Dates = list(Canada_deaths.columns.values)

Dates_dataf = pd.DataFrame(Dates)

Dates_dataf
total_death = list(Canada_deaths.T['Total Deaths'])

total_death = pd.DataFrame(total_death)

total_death
deaths_per_dates = pd.concat([Dates_dataf, total_death], axis =1)

deaths_per_dates.columns =['ds', 'y']

deaths_per_dates
Canada_data = covid_confirm.loc[covid_confirm['Country/Region']=='Canada'].iloc[:, 4:].sum(axis=0)

Canada_data
plot_data = range(len(Canada_data))

plt.bar(plot_data, Canada_data)

plt.title('Total Confirmed Cases of Canada')

plt.xlabel('Number of Days')

plt.ylabel('Number of Cases')

plt.show()
Canada_cases_per_day = Canada_data.diff()

print(Canada_cases_per_day)
Canada_cases_per_day = Canada_cases_per_day.fillna(Canada_data[0])

Canada_cases_per_day
plot2_data =range(len(Canada_cases_per_day))

plt.bar(plot2_data,Canada_cases_per_day)

plt.title('Daily Cases in Canada')

plt.xlabel('Number of Days')

plt.ylabel('Number of Cases')

plt.show()