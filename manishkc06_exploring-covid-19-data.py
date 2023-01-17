# Loading Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
# Load data

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
# A look at the dataset

covid_19_data.head()
# A concise summary of dataset

covid_19_data.info()
# Set the SNo as index

covid_19_data.set_index('SNo', inplace = True)
# Checking the shape 

covid_19_data.shape
# Renaming columns

covid_19_data.rename(columns={'Province/State': 'State', 'Country/Region': 'Country', 'Last Update': 'Last_Update'}, 

                     inplace=True)
# Check the columns

covid_19_data.columns
# Converting 'ObservationDate' and 'Last_Update' to datetime

covid_19_data.ObservationDate = pd.DatetimeIndex(covid_19_data.ObservationDate)

covid_19_data.Last_Update = pd.DatetimeIndex(covid_19_data.Last_Update)
# Renaming 'Mainland China' to 'Chine'

covid_19_data.Country = covid_19_data.Country.apply(lambda x: 'China' if x == 'Mainland China' else x)
# List of all the countries infected

countries = covid_19_data.Country.unique()

print(countries)

print()

print('Total number of countries infected: ', len(countries))
covid_19_data.Last_Update.quantile(1)
# The data was updated last on 13 June 2020.

updated_data = covid_19_data[covid_19_data.Last_Update == covid_19_data.Last_Update.quantile(1)]
# Visualizing the confirmed, deaths and recovered cases per country for top 10 countries with confirmed cases

fig = updated_data.groupby('Country').sum().sort_values(by = 'Confirmed', ascending = False)[:10].plot(

                                                                                            kind = 'bar',

                                                                                            figsize = (16, 8))



plt.ylabel('Number of cases')

plt.title("Top 10 countries with confirmed cases")



# To diplay the the count on top of the bar

for p in fig.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    fig.annotate('{:.0}'.format(height), (x, y + height + 0.01))
# percentage confirmed cases 

conf_cases = updated_data.groupby('Country')['Confirmed'].sum().sort_values(ascending = False) / updated_data['Confirmed'].sum()
# Plotting the percentage of confirmed cases worldwide

def value(val):

    per = np.round(val , 2)

    return per

conf_cases.plot(kind = 'pie', figsize = (16, 12), autopct = value)

plt.title('Percentage of confirmed cases worldwide')

plt.show()
# Getting top10 countries with highest number of confirmed cases

top_10 = updated_data[['Country', 'Confirmed','Recovered','Deaths']].groupby('Country').sum().sort_values(by = 'Confirmed', ascending = False)[:10]



# Recovery and Deaths percentage

top_10['Recovered_percentage'] = top_10['Recovered'] / top_10['Confirmed'] * 100

top_10['Deaths_percentage'] = top_10['Deaths'] / top_10['Confirmed'] * 100
top_10
top_10.Recovered_percentage.sort_values(ascending = False)
top_10.Deaths_percentage.sort_values(ascending=False)
top_10[['Recovered_percentage','Deaths_percentage']].plot(kind = 'bar', figsize = (16, 8))

plt.title("Recovered percentage and Deaths percentage of Top 10 Countries in terms of Confirmed Cases")

plt.ylabel("Percentage")

plt.show()
# USA data

USA_data = updated_data[updated_data.Country == 'US']

USA_data_by_state = USA_data[['State', 'Confirmed', 'Recovered', 'Deaths']].groupby('State').sum().sort_values(

                                                                                by = 'Confirmed',

                                                                                ascending = False)



USA_day_wise = covid_19_data[covid_19_data.Country == 'US'][['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']].groupby('ObservationDate').sum().sort_values(

                                                                                by = 'ObservationDate',

                                                                                ascending = True)
USA_total_recovered = USA_data_by_state.loc['Recovered']

USA_data_by_state = USA_data_by_state.drop('Recovered', axis = 0)
USA_total_recovered
# For USA overall recovered is given. Statewise recovered cases are not given in the dataset.

USA_data_by_state.plot(kind = 'bar', figsize = (16, 4))

plt.title("Statewise Confirmed and Deaths cases of USA")

plt.ylabel('Number of Cases')

plt.show()
USA_day_wise[['Confirmed', 'Deaths']].plot(figsize = (8,4))

plt.title('Confirmed and Deaths rate of USA')

plt.ylabel('Number of Cases')

plt.show()
# Germany data

Germany_data = updated_data[updated_data.Country == 'Germany']

Germany_data_by_state = Germany_data[['State', 'Confirmed', 'Recovered', 'Deaths']].groupby('State').sum().sort_values(

                                                                                by = 'Confirmed',

                                                                                ascending = False)



Germany_day_wise = covid_19_data[covid_19_data.Country == 'Germany'][['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']].groupby('ObservationDate').sum().sort_values(

                                                                                by = 'ObservationDate',

                                                                                ascending = True)
Germany_data_by_state.plot(kind = 'bar', figsize = (12,6))

plt.title("Covid data cases of Germany statewise")

plt.ylabel('Number of Cases')

plt.show()
Germany_day_wise.plot(figsize = (16,6))

plt.title("Confirmed, Recovered and Deaths rates of Germany per day")

plt.ylabel("Number of cases")

plt.show()
Russia_data = updated_data[updated_data.Country == 'Russia']

Russia_data_by_state = Russia_data[['State', 'Confirmed', 'Recovered', 'Deaths']].groupby('State').sum().sort_values(

                                                                                by = 'Confirmed',

                                                                                ascending = False)



Russia_day_wise = covid_19_data[covid_19_data.Country == 'Russia'][['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']].groupby('ObservationDate').sum().sort_values(

                                                                                by = 'ObservationDate',

                                                                                ascending = True)
Russia_data_by_state.plot(kind = 'bar', figsize = (15,6))

plt.title("Statewise Confirmed, Recovered and Deaths percentage of Russia")

plt.ylabel("Number of Cases")

plt.show()
Russia_day_wise.plot(figsize = (16, 6))

plt.title("Daywise Confirmed, Recovered and Deaths rates of Russia")

plt.ylabel('Number of Cases')

plt.show()
India_data = updated_data[updated_data.Country == 'India']

India_data_by_state = India_data[['State', 'Confirmed', 'Recovered', 'Deaths']].groupby('State').sum().sort_values(

                                                                                by = 'Confirmed',

                                                                                ascending = False)



India_day_wise = covid_19_data[covid_19_data.Country == 'India'][['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']].groupby('ObservationDate').sum().sort_values(

                                                                                by = 'ObservationDate',

                                                                                ascending = True)
India_data_by_state.plot(kind = 'bar', figsize = (16, 6))

plt.title("Statewise Confirmed, Recovered and Deaths cases in India")

plt.ylabel("Number of Cases")

plt.show()
India_day_wise.plot(figsize = (16, 6))

plt.title("Confirmed, Recovered and Deaths Rate in India per Day")

plt.ylabel("Number of Cases")

plt.show()