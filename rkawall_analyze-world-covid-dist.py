# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# read total covid tests performed by country
airport_traffic_df = pd.read_csv('/kaggle/input/uncover/UNCOVER/geotab/airport-traffic-analysis.csv')
world_dist_covid19_df = pd.read_csv('/kaggle/input/uncover/UNCOVER/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv')
hospital_beds_df = pd.read_csv('/kaggle/input/uncover/UNCOVER/esri_covid-19/esri_covid-19/definitive-healthcare-usa-hospital-beds.csv')
hospital_beds_df = hospital_beds_df.sort_values(["state_name", "county_nam"], ascending = (True, True))
hospital_beds_df.head()
world_dist_covid19_df.head()
world_dist_covid19_df['daterep'].unique()
countries_affected = world_dist_covid19_df['countriesandterritories'].unique()
all_countries_affected = list(countries_affected)
all_countries_affected
united_states_df = world_dist_covid19_df.loc[world_dist_covid19_df['countriesandterritories'] == 'United_States_of_America']
ax = plt.gca()

united_states_df.plot(kind='line',x='daterep',y='cases',ax=ax)
united_states_df.plot(kind='line',x='daterep',y='deaths', color='red', ax=ax)

plt.show()
most_deaths_reported_by_us = united_states_df['deaths'].max()
most_deaths_reported_by_us
# china reported 244 deaths on 2020-02-13
# we should check the number of cases in the us on that day and what the death toll was

feb_02_df = united_states_df.loc[united_states_df['daterep'] == '2020-02-13']
feb_02_df
most_deaths_recorded_by_day_us = united_states_df.loc[united_states_df['deaths'] == 1344]
most_deaths_recorded_by_day_us['daterep']
most_deaths_recorded_by_day_us
pandemic_declaration_us_df = united_states_df.loc[united_states_df['daterep'] == '2020-03-11']
pandemic_declaration_us_df
united_states_df['daterep'] = pd.to_datetime(united_states_df['daterep'])
start_date = '2020-02-13'
end_date = '2020-03-12'

mask = (united_states_df['daterep'] > start_date) & (united_states_df['daterep'] <= end_date)

us_0213_0311_df = united_states_df.loc[mask]
us_0213_0311_df.head()
population_0213_0311 = list(us_0213_0311_df['popdata2018'])[0]
num_infections_0213_0311 = us_0213_0311_df['cases'].sum()
k = 100

infecttion_rate_0213_0311 = (num_infections_0213_0311 / population_0213_0311) * k

infecttion_rate_0213_0311
num_infections_0213_0311
num_deaths_0213_0311 = us_0213_0311_df['deaths'].sum()
num_deaths_0213_0311
# infection rate in the US

population = list(united_states_df['popdata2018'])[0] # population is 327167434.0
num_infections = united_states_df['cases'].sum() # num_infections is 312237
k = 100 

infection_rate = (num_infections / population) * k

infection_rate # infection_rate is 0.09543645471755603

# infection rate of 9.54 % as of 2020-04-05
# from 04/05/2020 to present the numbers of cases went up by 787,763 (23 days) This is due to mainly more testing

from sklearn.linear_model import LinearRegression
us_dataset = united_states_df[['cases', 'deaths']]

X = us_dataset.iloc[:, 0].values.reshape(-1, 1)
Y = us_dataset.iloc[:, 1].values.reshape(-1, 1)

linear_regressor = LinearRegression()

linear_regressor.fit(X, Y)  # perform linear regression

Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
china_df = world_dist_covid19_df.loc[world_dist_covid19_df['countriesandterritories'] == 'China']

ax = plt.gca()

china_df.plot(kind='line',x='daterep',y='cases',ax=ax)
china_df.plot(kind='line',x='daterep',y='deaths', color='red', ax=ax)

plt.show()
most_deaths_reported_by_china = china_df['deaths'].max()

most_deaths_reported_by_china
most_deaths_recorded_by_day_china = china_df.loc[china_df['deaths'] == 254]
most_deaths_recorded_by_day_china['daterep']
# infection rate in the china

population = list(china_df['popdata2018'])[0] # population is 1392730000.0
num_infections = china_df['cases'].sum() # num_infections is 82575
k = 100 

infection_rate = (num_infections / population) * k

infection_rate # infection_rate is 0.005929002749994615

# infection_rate is 0.59 % per 100 
copy_world_dist_covid19_df = world_dist_covid19_df
all_countries_affected_before_pandemic = pd.to_datetime(copy_world_dist_covid19_df['daterep'])
overall_end_date = '2020-03-12'

all_mask = (copy_world_dist_covid19_df['daterep'] <= end_date)

all_countries_0311_df = copy_world_dist_covid19_df.loc[all_mask]
all_countries_0311_df.head()
num_cases_world = all_countries_0311_df['cases'].sum()
num_cases_world
max_cases = all_countries_0311_df['cases'].max()
max_cases
country_with_most_cases_before_pandemic = all_countries_0311_df.loc[all_countries_0311_df['cases'] == max_cases]
country_with_most_cases_before_pandemic
all_countries_excluding_china_df = all_countries_0311_df.loc[all_countries_0311_df['countriesandterritories'] != 'China']
all_countries_excluding_china_df.head()
max_cases_excluding_china = all_countries_excluding_china_df['cases'].max()
max_cases_excluding_china
country_with_most_cases_before_pandemic_excluding_china = all_countries_excluding_china_df.loc[all_countries_excluding_china_df['cases'] == max_cases_excluding_china]
country_with_most_cases_before_pandemic_excluding_china