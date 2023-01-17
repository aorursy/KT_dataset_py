# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import matplotlib.ticker as ticker
import scipy.interpolate as ip
import calendar
import datetime

# Data Read From File
df_airline_portcountry = pd.read_csv('/kaggle/input/airline-by-country-of-port-data/airline_portcountry.csv', delimiter=',')
#Airport Traffic data File
df_mon_acm_web = pd.read_csv('/kaggle/input/airport-traffic-data/mon_acm_web.csv', delimiter=',')
df_mon_pax_web = pd.read_csv('/kaggle/input/airport-traffic-data/mon_pax_web.csv', delimiter=',')
df_yed_acm_web = pd.read_csv('/kaggle/input/airport-traffic-data/yed_acm_web.csv', delimiter=',')
df_yed_pax_web = pd.read_csv('/kaggle/input/airport-traffic-data/yed_pax_web.csv', delimiter=',')
df_yej_acm_web = pd.read_csv('/kaggle/input/airport-traffic-data/yej_acm_web.csv', delimiter=',')
df_yej_pax_web = pd.read_csv('/kaggle/input/airport-traffic-data/yej_pax_web.csv', delimiter=',')

# Domestic Airlines Top Routes and Totals
df_dom_citypairs_web = pd.read_csv('/kaggle/input/domestic-airlines-top-routes-and-totals/dom_citypairs_web.csv', delimiter=',')
df_dom_totals_web = pd.read_csv('/kaggle/input/domestic-airlines-top-routes-and-totals/dom_totals_web.csv', delimiter=',')

# International Airline traffic by City Pairs
df_city_pairs = pd.read_csv('/kaggle/input/international-airlines-traffic-by-city-pairs/city_pairs.csv', delimiter=',')

#Removing Duplicate TOTAL AUSTRALIA row
df_yed_acm_web = df_yed_acm_web[df_yed_acm_web['AIRPORT']!= 'TOTAL AUSTRALIA']
df_yej_pax_web = df_yej_pax_web[df_yej_pax_web['AIRPORT']!= 'TOTAL AUSTRALIA']
df_yej_acm_web = df_yej_acm_web[df_yej_acm_web['AIRPORT']!= 'TOTAL AUSTRALIA']
df_yej_pax_web = df_yej_pax_web[df_yej_pax_web['AIRPORT']!= 'TOTAL AUSTRALIA']

#Filtering By Year
df_airline_portcountry =df_airline_portcountry[df_airline_portcountry['Year']<2020]
df_mon_acm_web =df_mon_acm_web[df_mon_acm_web['Year']<2020]
df_mon_pax_web =df_mon_pax_web[df_mon_pax_web['Year']<2020]
df_yed_acm_web =df_yed_acm_web[df_yed_acm_web['Year_Ended_December']<2020]
df_yed_pax_web =df_yed_pax_web[df_yed_pax_web['Year_Ended_December']<2020]
df_yej_acm_web =df_yej_acm_web[df_yej_acm_web['Year_Ended_June']<2020]
df_yej_pax_web =df_yej_pax_web[df_yej_pax_web['Year_Ended_June']<2020]
df_dom_citypairs_web =df_dom_citypairs_web[df_dom_citypairs_web['Year']<2020]
df_dom_totals_web =df_dom_totals_web[df_dom_totals_web['Year']<2020]
df_city_pairs =df_city_pairs[df_city_pairs['Year']<2020]

#Rename Columns
df_airline_portcountry.rename(columns={'Passengers_In': 'PassengersIn'}, inplace=True)

# Calculate the %
def total_contribution(value):
    return value*100/tot_passengers


# Identify if the data is considered as Others or not
def country_name(name):
    if name in popular_passenger_countries:
        return name
    else:
        return 'Others'
df_numberof_int_flights = df_yed_acm_web.groupby('Year_Ended_December')['Int_Acm_Total'].sum()
df_numberof_dom_flights = df_dom_totals_web.groupby('Year')['Aircraft_Trips'].sum()
df_total_intl_flights = df_numberof_int_flights.reset_index().rename(columns={'Year_Ended_December':'Year', 'Int_Acm_Total': 'Intl_Trips'})
df_total_dom_flights = df_numberof_dom_flights.reset_index().rename(columns={'Aircraft_Trips':'Dom_Trips'})
df_total_flights = df_total_dom_flights.merge(df_total_intl_flights, how='inner', on='Year')
df_total_flights['Domestic'] = df_total_flights['Dom_Trips']*100/(df_total_flights['Dom_Trips'] + df_total_flights['Intl_Trips'])
df_total_flights['International'] = df_total_flights['Intl_Trips']*100/(df_total_flights['Dom_Trips'] + df_total_flights['Intl_Trips'])
df_total_flights.drop(['Dom_Trips', 'Intl_Trips'], axis=1, inplace=True)
df_total_flights.set_index('Year', inplace=True)
    
#Plot Graph
ax = df_total_flights.plot(
    kind='bar', 
    stacked=True,
    color=('#5cb85c', '#5bc0de'),
    figsize=(16, 6), 
    legend=True,
    xlim=(1985, 2019),
    ylim=(60,102),
    fontsize=14
)

ax.set_title("Domestic vs International Flights from 1985-2019", fontsize=16)
ax.legend(loc='upper right', frameon=True, fontsize=14, bbox_to_anchor=(1.17, 1.0))
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Flights %", fontsize=14)
plt.xticks()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(b=True, linewidth=0.5, axis='y');
df_airline_portcountry['Total'] = df_airline_portcountry['PassengersIn']+df_airline_portcountry['Passengers_Out']
df_airline_portcountry_3 = df_airline_portcountry[df_airline_portcountry['Airline']=='Qantas Airways']
df_airline_portcountry_3 = df_airline_portcountry_3[['Port_Country', 'Total', 'Year']]
df_airline_portcountry_3 = df_airline_portcountry_3[df_airline_portcountry_3['Year'] > 2014]

piv_country = df_airline_portcountry_3.groupby(['Port_Country'])[['Total']].sum()
piv_country = piv_country.sort_values(['Total'])
piv_country = piv_country.tail(5)
popular_passenger_countries = piv_country.index

mask_1 = df_airline_portcountry_3['Port_Country'].isin(popular_passenger_countries)
df_airline_portcountry_3=df_airline_portcountry_3[mask_1]

piv_country = df_airline_portcountry_3.groupby(['Year', 'Port_Country'])[['Total']].sum()
piv_country = df_airline_portcountry_3.pivot_table(index = 'Year', columns= 'Port_Country', aggfunc='sum')
piv_country.columns = piv_country.columns.get_level_values(1)
piv_country

ax = piv_country.plot(
    kind='line', 
    figsize=(16, 6), 
    legend=True,
    fontsize=14,
    marker='D',
    linewidth =4.8
)
ax.set_title("Passengers by Country from 2015-2019", fontsize=16)

ax.legend(loc='lower center', frameon=True, fontsize=14, ncol=5, bbox_to_anchor=(0.45, -0.25))
ax.set_ylabel("Passengers in Million", fontsize=14)
ax.set_xlabel("Year", fontsize=14)
ax.yaxis.set_major_formatter(ticker.EngFormatter())
plt.xticks(piv_country.index)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(b=True, linewidth=0.5, axis='y')
df_airline_portcountry_3 = df_airline_portcountry[df_airline_portcountry['Port_Country']=='New Zealand']
df_airline_portcountry_3

piv_month = df_airline_portcountry_3.groupby(['Month_num'])[['Total']].sum()

piv_month

ax = piv_month.plot(
    kind='line', 
    figsize=(16, 6), 
    legend=True,
    fontsize=14,
    marker='D',
    linewidth =4.8
)
ax.set_title("Passengers Travelling between New Zealand and Australia from 2015 to 2019 Monthly", fontsize=16)

ax.legend(['Number of Passengers'], loc='lower center', frameon=True, fontsize=14, ncol=5, bbox_to_anchor=(0.45, -0.25))
ax.set_ylabel("Passengers in Million", fontsize=14)
ax.set_xlabel("Year", fontsize=14)
ax.yaxis.set_major_formatter(ticker.EngFormatter())
ax.set_xticklabels(['Jan','Feb', 'Apr','Jun', 'Aug', 'Oct','Dec'])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(b=True, linewidth=0.5, axis='y')
df_airline_portcountry['Total_Freight'] = df_airline_portcountry['Freight_In_(tonnes)']+df_airline_portcountry['Freight_Out_(tonnes)']
+df_airline_portcountry['Mail_In_(tonnes)']+df_airline_portcountry['Mail_Out_(tonnes)']
df_airline_country = df_airline_portcountry[df_airline_portcountry['Year']>2014]
piv_freight = df_airline_country.groupby(['Port_Country'])[['Total_Freight']].sum()
piv_freight = piv_freight.sort_values(['Total_Freight'])
piv_freight = piv_freight.tail(5)
popular_freight_countries = piv_freight.index

df_airline_country = df_airline_country[['Year', 'Port_Country', 'Total_Freight']]
mask_1 = df_airline_country['Port_Country'].isin(popular_freight_countries)
df_freight_by_country=df_airline_country[mask_1]

piv_country = df_freight_by_country.groupby(['Year', 'Port_Country'])[['Total_Freight']].sum()
piv_country = df_freight_by_country.pivot_table(index = 'Year', columns= 'Port_Country', aggfunc='sum')

piv_country.columns = piv_country.columns.get_level_values(1)

ax = piv_country.plot(
    kind='line', 
    figsize=(16, 6), 
    legend=True,
    fontsize=14,
    marker='D',
    linewidth =4.8
)
ax.set_title("Freights and Mails by Country from 2015-2019", fontsize=16)

ax.legend(loc='lower center', frameon=True, fontsize=14, ncol=5, bbox_to_anchor=(0.45, -0.25))
ax.set_ylabel("Freights (in Tonnes)", fontsize=14)
ax.set_xlabel("Year", fontsize=14)
ax.yaxis.set_major_formatter(ticker.EngFormatter())
plt.xticks(piv_country.index)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(b=True, linewidth=0.5, axis='y')
df_airline_portcountry_2 = df_airline_portcountry[df_airline_portcountry['Year']>2014]
df_airline_portcountry_3 = df_airline_portcountry_2[['Airline', 'Total', 'Year']]

piv_pass = df_airline_portcountry_2.groupby(['Airline'])[['Total']].sum()
piv_pass = piv_pass.sort_values(['Total'])
piv_pass = piv_pass.tail(5)
airlines = piv_pass.index

mask_1 = df_airline_portcountry_3['Airline'].isin(airlines)
df_airline_portcountry_3=df_airline_portcountry_3[mask_1]

piv_pass = df_airline_portcountry_3.groupby(['Year', 'Airline'])[['Total']].sum()
piv_pass = df_airline_portcountry_3.pivot_table(index = 'Year', columns= 'Airline', aggfunc='sum')

piv_pass.columns = piv_pass.columns.get_level_values(1)

ax = piv_pass.plot(
    kind='line', 
    figsize=(16, 6), 
    marker='D',
    legend=True,
    fontsize=14,
    linewidth=4.8
)
ax.set_title("Passengers by Airlines from 2015-2019", fontsize=16)
ax.legend(loc='center right', frameon=True, fontsize=14)
ax.set_ylabel("Passengers in Million", fontsize=14)
ax.set_xlabel("Year", fontsize=14)
ax.yaxis.set_major_formatter(ticker.EngFormatter())
plt.xticks(piv_pass.index)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(b=True, linewidth=0.5, axis='y')
df_airline_portcountry_3 = df_airline_portcountry[['Airline', 'Total', 'Year']]
df_airline_portcountry_3 = df_airline_portcountry_3[df_airline_portcountry_3['Year'] > 2014]
tot_passengers = df_airline_portcountry_3['Total'].sum()

piv_country = df_airline_portcountry_3.groupby(['Airline'])[['Total']].sum()
piv_country = piv_country.sort_values(['Total'])
popular_passenger_countries = piv_country.tail(7).index

cat_df = piv_country.reset_index()

cat_df['airlines'] = cat_df['Airline'].apply(country_name)
piv_country = cat_df.groupby(['airlines'])[['Total']].sum()
piv_country['Total_Contribution'] = piv_country['Total'].apply(total_contribution)
piv_country = piv_country.sort_values(['Total'], ascending=False)

my_labels = piv_country['Total_Contribution']
plot = piv_country['Total_Contribution'].plot.pie(figsize=(8, 6), ylabel='', autopct='%1.1f%%', title = 'Airlines 2015-2019')


df_airline_portcountry_3 = df_airline_portcountry[['Port_Country', 'Total', 'Year']]
df_airline_portcountry_3 = df_airline_portcountry_3[df_airline_portcountry_3['Year'] > 2014]
tot_passengers = df_airline_portcountry_3['Total'].sum()

piv_country = df_airline_portcountry_3.groupby(['Port_Country'])[['Total']].sum()
piv_country = piv_country.sort_values(['Total'])
popular_passenger_countries = piv_country.tail(7).index

cat_df = piv_country.reset_index()

cat_df['country'] = cat_df['Port_Country'].apply(country_name)
piv_country = cat_df.groupby(['country'])[['Total']].sum()
piv_country['Total_Contribution'] = piv_country['Total'].apply(total_contribution)
piv_country = piv_country.sort_values(['Total'], ascending=False)

my_labels = piv_country['Total_Contribution']
plot = piv_country['Total_Contribution'].plot.pie(figsize=(8, 6), ylabel='',autopct='%1.1f%%', title ='Passengers by Country 2015-2019')

df_airline_portcountry_3 = df_airline_portcountry[['Port_Country', 'Total_Freight', 'Year']]
df_airline_portcountry_3 = df_airline_portcountry_3[df_airline_portcountry_3['Year'] > 2014]
tot_passengers = df_airline_portcountry_3['Total_Freight'].sum()

piv_country = df_airline_portcountry_3.groupby(['Port_Country'])[['Total_Freight']].sum()
piv_country = piv_country.sort_values(['Total_Freight'])
popular_passenger_countries = piv_country.tail(7).index

cat_df = piv_country.reset_index()

cat_df['country'] = cat_df['Port_Country'].apply(country_name)
piv_country = cat_df.groupby(['country'])[['Total_Freight']].sum()
piv_country['Total_Contribution'] = piv_country['Total_Freight'].apply(total_contribution)
piv_country = piv_country.sort_values(['Total_Freight'], ascending=False)

my_labels = piv_country['Total_Contribution']
plot = piv_country['Total_Contribution'].plot.pie(figsize=(8, 6), ylabel='', autopct='%1.1f%%', title = 'Freight distribution 2015-2019')
