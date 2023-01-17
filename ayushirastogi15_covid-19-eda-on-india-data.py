# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.graph_objects as go

import matplotlib.pyplot as plt

import folium
# 'covid_19_india.csv' contains cured, deaths and confirmed cases on a day-to-day basis.

cases = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

cases.head()



# It contains latitude and longitude coordinates of indian states.

lat_long = pd.read_csv("/kaggle/input/lat-lon-indian-states/datasets_652259_1374178_Total_India_covid-19.csv")

lat_long.head()



# It can provide us the ratio or rate or percentage of positive cases per population statewise.

popul = pd.read_csv("/kaggle/input/covid19-in-india/population_india_census2011.csv")

popul
# No null value or no null record is found.

cases.info()
cases['State/UnionTerritory'].unique()
cases['State/UnionTerritory'].replace({"Telengana" : "Telangana", "Telengana***" : "Telangana",

                                        "Telangana***" : "Telangana"}, inplace = True)



cases['State/UnionTerritory'].replace({"Daman & Diu" : "Dadra and Nagar Haveli and Daman and Diu",

                                          "Dadar Nagar Haveli" : "Dadra and Nagar Haveli and Daman and Diu"},

                                         inplace = True)

cases = cases[(cases['State/UnionTerritory'] != 'Unassigned') &

                    (cases['State/UnionTerritory'] != 'Cases being reassigned to states')]

cases['State/UnionTerritory'].unique()
popul.rename(columns={'State / Union Territory':'State/UnionTerritory'}, inplace=True)

popul['State/UnionTerritory'].replace({"Telengana" : "Telangana"})
cases.Date = pd.to_datetime(cases.Date, dayfirst=True)



cases.drop(['Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational'], axis = 1, inplace=True)

cases.head()
print("Starting date : ", min(cases.Date.values))

print("Ending date : ", max(cases.Date.values))
daily_cases = cases.groupby('Date').sum().reset_index()

daily_cases['Active'] = 1



for val in daily_cases.index:

    if val != 0:

        daily_cases['Active'].loc[val] = daily_cases['Confirmed'].loc[val] - daily_cases['Cured'].loc[val-1] - daily_cases['Deaths'].loc[val-1]

    

daily_cases
fig = go.Figure()

fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Confirmed, name = 'Confirmed'))

fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Cured, name = 'Cured'))

fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Deaths, name = 'Deaths'))

fig.add_trace(go.Scatter(x = daily_cases.Date, y = daily_cases.Active, name = 'Active Cases'))



fig.update_layout(title = 'CORONA VIRUS CASES IN INDIA', yaxis_title = 'Cases Count (in lakhs)')



fig.show()
country_cases = cases[cases.Date == max(cases.Date)]

print(country_cases.shape)

country_cases.head()
lat_long.rename(columns={"State":"State/UnionTerritory"}, inplace=True)

lat_long = lat_long[['State/UnionTerritory', 'Latitude', 'Longitude']]

country_cases = pd.merge(country_cases, lat_long, on='State/UnionTerritory')

country_cases.head()
# Make an empty map

m = folium.Map(location=[28,77], zoom_start=4)

country_cases['Confirmed'] = country_cases['Confirmed'].astype(float)



# I can add marker one by one on the map

for i in range(0,len(country_cases)):

    folium.Circle(location = [country_cases.iloc[i]['Latitude'], country_cases.iloc[i]['Longitude']],

                popup = [country_cases.iloc[i]['State/UnionTerritory'],country_cases.iloc[i]['Confirmed']],

                radius = country_cases.iloc[i]['Confirmed']/2,

                color = 'crimson', fill = True, fill_color='crimson').add_to(m)

    

m
# We're saving it for further use

total_pop = popul['Population'].sum()

print("The total population of India is : ", total_pop)



popul = cases.merge(popul[['State/UnionTerritory', 'Population']])

popul['ConfirmPerc'] = 0

popul['ConfirmPerc'] = (popul['Confirmed']/popul['Population'])*100
fig = go.Figure()

for st in popul['State/UnionTerritory'].unique():

    df = popul[popul['State/UnionTerritory'] == st]

    fig.add_trace(go.Scatter(x = df.Date, y = df.ConfirmPerc, name = st))

    

fig.update_layout(title = 'Positive Cases Percentage Per Population', yaxis_title = 'Percentage (%)')

fig.show()
# Here, we're grouping the data by date bcoz we want to visualize the data on per basis

popul_dates = popul.drop('ConfirmPerc', axis=1).groupby('Date').sum()



# In this, population should be same, as we're talking about the positive percentage per population

popul_dates['Population'] = total_pop



# Calculating total percentage of positive cases

popul_dates['TotConfirmPerc'] = (popul_dates['Confirmed']/popul_dates['Population'])*100

popul_dates
fig = go.Figure()

fig.add_trace(go.Scatter(x = popul_dates.index, y = popul_dates.TotConfirmPerc))

fig.update_layout(title = 'Percentage of positive cases across India', yaxis_title = 'Percentage (%)')

fig.show()