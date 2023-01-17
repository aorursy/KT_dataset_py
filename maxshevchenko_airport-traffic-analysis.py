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

import plotly.express as px



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



df_airline_portcountry.rename(columns={'Passengers_In': 'PassengersIn'}, inplace=True)





df_numberof_dom_flights = df_dom_totals_web.groupby('Year')['Aircraft_Trips'].sum()

df_numberof_dom_flights.head(12)

#Plot Graph

px.line(df_numberof_dom_flights,y='Aircraft_Trips')
df_numberof_int_flights = df_yed_acm_web.groupby('Year_Ended_December')['Int_Acm_Total'].sum()



#Plot Graph

px.line(df_numberof_int_flights,y='Int_Acm_Total')
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

    ylim=(50,102),

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
#df_airline_portcountry = df_dom_totals_web.groupby('Year')['Passengers_Out'].sum()

piv_passengersIn = pd.pivot_table(data=df_airline_portcountry, values='PassengersIn', index=['Year'],aggfunc='sum')

ax = piv_passengersIn.plot(

    kind='line', 

    color=('#5cb85c', '#5bc0de'),

    figsize=(16, 6), 

    legend=True,

    xlim=(1985, 2019),

    fontsize=14

    ,linewidth =4.8

)



ax.set_title("Number of passengers coming into Australia from 1985-2019", fontsize=16)

ax.legend(['Inbound Passengers'], loc='center right', frameon=True, fontsize=14)

ax.set_ylabel("Number of Passenegers", fontsize=14)

ax.set_xlabel("Year", fontsize=14)

ax.yaxis.set_major_formatter(ticker.EngFormatter())

plt.xticks()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

plt.grid(b=True, linewidth=0.5, axis='y');

#####Problem in Million value plotting in yAxis Label
piv_passengersOut = pd.pivot_table(data=df_airline_portcountry, values='Passengers_Out', index=['Year'],aggfunc='sum')

ax = piv_passengersOut.plot(

    kind='line', 

    color=('#5cb85c', '#5bc0de'),

    figsize=(16, 6), 

    legend=True,

    xlim=(1985, 2019),

    fontsize=14

    ,linewidth =4.8

)



ax.set_title("Number of Passengers going out of Australia from 1985-2019", fontsize=16)

ax.legend(['Outbound Passengers'], loc='center right', frameon=True, fontsize=14)

ax.set_ylabel("Number of Passenegers", fontsize=14)

ax.set_xlabel("Year", fontsize=14)

ax.yaxis.set_major_formatter(ticker.EngFormatter())

plt.xticks()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

plt.grid(b=True, linewidth=0.5, axis='y')
df_airline_portcountry['Total'] = df_airline_portcountry['PassengersIn']+df_airline_portcountry['Passengers_Out']

df_portCountry_pOut = df_airline_portcountry.groupby('Port_Country')[['PassengersIn','Passengers_Out','Total']].sum()

df_portCountry_pOut.reset_index(inplace=True)

mask = df_portCountry_pOut['Port_Country'].isin(['New Zealand','Papua New Guinea', 'Fiji','Solomon Islands','Vanuatu'])

df_portCountry_pOut =df_portCountry_pOut[mask]

df_portCountry_pOut.head(10)

df_portCountry_pOut.drop(['PassengersIn', 'Passengers_Out'], axis=1, inplace=True)

df_portCountry_pOut.set_index('Port_Country', inplace=True)

df_portCountry_pOut = df_portCountry_pOut.sort_values('Total', ascending=False)



ax = df_portCountry_pOut.plot(

    kind='barh', 

    color=('#5cb85c', '#5bc0de'),

    figsize=(16, 6), 

    legend=False,

    fontsize=14

    ,linewidth =4.8

)



ax.set_title("Passengers traveling between Australia and Oceania from 1985-2019", fontsize=16)

# ax.legend(loc='upper right', frameon=True, fontsize=14)

ax.set_ylabel('Country', fontsize=14)

ax.set_xlabel("Number of Passenegers", fontsize=14)

# ax.set_xlabel(['{:,.0f}'.format(x) + 'K' for x in ax.get_xticks()/1000])

ax.xaxis.set_major_formatter(ticker.EngFormatter())

plt.xticks()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

plt.grid(b=True, linewidth=0.5, axis='y')
piv_Dom_Least_fav =df_yed_pax_web.groupby('AIRPORT')['Dom_Pax_Total'].sum()

piv_Dom_Least_fav = piv_Dom_Least_fav.sort_values()

piv_Dom_Least_fav= piv_Dom_Least_fav.head(10)



ax = piv_Dom_Least_fav.plot(

    kind='bar', 

    color=('#5cb85c'),

    figsize=(16, 6), 

    legend=True,

    xlim=(1985, 2019),

    fontsize=14

    ,linewidth =4.8

)



ax.set_title("Least Favourite Travel Destinantions for Domestic Flights from 1985-2019", fontsize=16)

ax.legend(['Domestic Passengers'],loc='upper right', frameon=True, fontsize=14, bbox_to_anchor=(1.25, 1.0))

ax.set_ylabel("Number Of Passenegers (In Millions)", fontsize=14)

ax.set_xlabel("City", fontsize=14)

ax.yaxis.set_major_formatter(ticker.EngFormatter())

plt.xticks()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

plt.grid(b=True, linewidth=0.5, axis='y')
piv_int_Least_fav =df_yed_pax_web.groupby('AIRPORT')[['Int_Pax_Total']].sum()

piv_int_Least_fav.reset_index(inplace=True)

piv_int_Least_fav = piv_int_Least_fav[piv_int_Least_fav['Int_Pax_Total']>0]

piv_int_Least_fav= piv_int_Least_fav.sort_values(['Int_Pax_Total'])

piv_int_Least_fav= piv_int_Least_fav.head(9)

piv_int_Least_fav= piv_int_Least_fav.set_index('AIRPORT')

ax = piv_int_Least_fav.plot(

    kind='bar', 

    color=('#5cb85c', '#5bc0de'),

    figsize=(16, 6), 

    legend=True,

    xlim=(1985, 2019),

    fontsize=14

    ,linewidth =4.8

)



ax.set_title("Least Favourite Travel Destinantions for International Flights from 1985-2019", fontsize=16)

ax.legend(['International Passengers'],loc='upper right', frameon=True, fontsize=14, bbox_to_anchor=(1.25, 1.0))

ax.set_ylabel("Number of Passenegers", fontsize=14)

ax.set_xlabel("City", fontsize=14)

ax.yaxis.set_major_formatter(ticker.EngFormatter())

plt.xticks()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

plt.grid(b=True, linewidth=0.5, axis='y')
piv_monthly_number =df_mon_pax_web.groupby('Month')['Pax_Total'].sum()

ax = piv_monthly_number.plot(

    kind='line', 

    color=('#5cb85c', '#5bc0de'),

    figsize=(16, 6), 

    legend=True,

    marker='D',

    markersize=8,

    xlim=(0.5, 12.5),

    fontsize=14

    ,linewidth =4.8

)



ax.set_title("Monthly number of Passengers from 1985-2019", fontsize=16)

ax.legend(['Passengers'], loc='upper right', frameon=True, fontsize=14)

ax.set_ylabel("Number of Passenegers", fontsize=14)

ax.set_xlabel("Months", fontsize=14)

ax.yaxis.set_major_formatter(ticker.EngFormatter())

plt.xticks()

ax.set_xticklabels(['Jan','Feb','Apr','Jun', 'Aug', 'Oct','Dec'])



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

plt.grid(b=True, linewidth=0.5, axis='y')

piv_max_freight =df_airline_portcountry.groupby('Port_Country')[['Freight_In_(tonnes)','Freight_Out_(tonnes)']].sum()

piv_max_freight = piv_max_freight.nlargest(10,'Freight_In_(tonnes)')

ax = piv_max_freight.plot(

    kind='bar', 

    color=('#5cb85c', '#5bc0de'),

    figsize=(16, 6), 

    legend=True,

    xlim=(0, 12),

    fontsize=14

    ,linewidth =4.8

)

ax.set_title("The Maximum Freight for Top 10 countries from 1985-2019", fontsize=16)

ax.legend(['Freight In', 'Freight Out'], loc='upper right', frameon=True, fontsize=14)

ax.set_ylabel("Weight (in tonnes)", fontsize=14)

ax.set_xlabel("Countries", fontsize=14)

ax.yaxis.set_major_formatter(ticker.EngFormatter())

plt.xticks()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

plt.grid(b=True, linewidth=0.5, axis='y')
piv_max_pass_freight =df_airline_portcountry.groupby('Airline')[['PassengersIn','Passengers_Out']].sum()

piv_max_pass_freight = piv_max_pass_freight.nlargest(10, 'PassengersIn')

ax = piv_max_pass_freight.plot(

    kind='bar', 

    color=('#5cb85c', '#5bc0de'),

    figsize=(16, 6), 

    legend=True,

    xlim=(0, 12),

    fontsize=14

    ,linewidth =4.8

)

ax.set_title("Top 10 Airlines from 1985-2019", fontsize=16)

ax.legend(['Inbound Passengers', 'Outbound Passengers'], loc='upper right', frameon=True, fontsize=14)

ax.set_ylabel("Number of Passengers", fontsize=14)

ax.set_xlabel("Airlines", fontsize=14)

ax.yaxis.set_major_formatter(ticker.EngFormatter())

plt.xticks()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

plt.grid(b=True, linewidth=0.5, axis='y')