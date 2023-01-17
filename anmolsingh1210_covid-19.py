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



import matplotlib.pyplot as plt

import seaborn as sns



from plotly import tools

import plotly.express as px
data = pd.read_csv('../input/covid.csv')

data.head()
data.info()
data = data.drop(['Unnamed: 0','Latitude','Longitude','FIPS','Admin2','Combined_Key'],axis=1)
data['Country_Region'] = data['Country_Region'].replace({'Mainland China':'China'})
data['Date'] = pd.to_datetime(data['Date'])

data['Date'] = data['Date'].dt.strftime('%m/%d/%Y')
data.info()
data.isnull().sum()
# getting number of active cases

data['Active'] = data['Confirmed'] - data['Deaths'] - data['Recovered']

data.head()
latest = data[data['Date'] == data['Date'].max()]

latest.head()
latest_data = latest.groupby('Country_Region')['Confirmed','Deaths','Recovered','Active'].sum().reset_index()

latest_data.head()
dates = data.groupby('Date')[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()

dates.head()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.8')

plt.plot(dates.Date, dates.Confirmed, color='blue', label='Confirmed')

plt.plot(dates.Date, dates.Deaths, color='red', label='Deaths')

plt.plot(dates.Date, dates.Recovered, color='green', label='Recovered')

plt.fill_between(dates.Date, dates.Confirmed, color='lightblue', alpha=1)

plt.fill_between(dates.Date, dates.Deaths, color='red', alpha=0.5)

plt.fill_between(dates.Date, dates.Recovered, color='lightgreen', alpha=0.7)

plt.xticks(rotation='90')

plt.xlabel('Dates', size=15)

plt.ylabel('Number of cases', size=15)

plt.title('Corona cases over time', size=15)

plt.legend()

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.8')

plt.plot(dates.Date, dates.Active, color='orange', label='Active cases')

plt.plot(dates.Date, dates.Recovered, color='green', label='Recovered')

plt.fill_between(dates.Date, dates.Active, color='yellow', alpha=0.5)

plt.fill_between(dates.Date, dates.Recovered, color='lightgreen', alpha=0.5)

plt.xticks(rotation='90')

plt.xlabel('Dates', size=15)

plt.ylabel('Number of cases', size=15)

plt.title('Active and Recovered cases over time', size=15)

plt.legend()

plt.show()
recovery_rate = np.round(dates['Recovered']/dates['Confirmed'], 3)*100

mortality_rate = np.round(dates['Deaths']/dates['Confirmed'], 3)*100



plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.8')

plt.plot(dates.Date, recovery_rate, color='orange', label='Recovered rate')

plt.plot(dates.Date, mortality_rate, color='green', label='Death rate')

plt.fill_between(dates.Date, recovery_rate, color='yellow', alpha=0.5)

plt.fill_between(dates.Date, mortality_rate, color='lightgreen', alpha=0.5)

plt.xticks(rotation='90')

plt.xlabel('Dates', size=15)

plt.ylabel('Percentage', size=15)

plt.title('Recovery and mortality rate cases over time', size=15)

plt.legend()

plt.show()
dates['New_cases'] = 0

for i in dates.index-1:

    dates['New_cases'].iloc[i] = dates['Confirmed'].iloc[i]-dates['Confirmed'].iloc[i-1]

dates['New_cases'].iloc[0] = dates['Confirmed'].iloc[0]

dates.head()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.8')

sns.barplot(dates.Date, dates.New_cases, palette='winter', edgecolor='k')

plt.xticks(rotation='90')

plt.title('Daily increase in cases', size=15)

plt.xlabel('Dates', size=15)

plt.ylabel('Number of cases', size=15)

plt.show()
temp = data.groupby(['Date','Country_Region'])[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()

temp.head()



temp['size'] = temp['Confirmed'].pow(0.3) * 3.5

px.scatter_geo(temp, locations='Country_Region',locationmode='country names', color='Confirmed', hover_name='Country_Region',

               size='size', range_color=[1,100], animation_frame='Date', projection='natural earth',

               color_continuous_scale='jet', title='Covid-19 cases over time').show()
Country = pd.DataFrame()

Country['Name'] = latest_data["Country_Region"]

Country['Values'] = latest_data["Confirmed"]

px.choropleth(Country, locations='Name', locationmode='country names', color="Values",

              title="Corona spread on 25-03-2020").show()
top20_countries = latest_data.sort_values('Confirmed',ascending=False).head(20).reset_index()

top20_countries = top20_countries.drop('index', axis=1)

top20_countries
plt.figure(figsize=(14,6))

plt.grid(True, color='grey')

plt.gca().patch.set_facecolor('0.8')

plt.plot(top20_countries['Country_Region'],top20_countries['Confirmed'], 'b.-', label='Confirmed')

plt.plot(top20_countries['Country_Region'],top20_countries['Recovered'], 'g.-', label='Recovered')

plt.plot(top20_countries['Country_Region'],top20_countries['Deaths'], 'r.-', label='Death')

plt.fill_between(top20_countries['Country_Region'],top20_countries['Confirmed'], color='lightblue', alpha=0.8)

plt.fill_between(top20_countries['Country_Region'],top20_countries['Recovered'], color='lightgreen', alpha=0.5)

plt.fill_between(top20_countries['Country_Region'],top20_countries['Deaths'], color='red', alpha=0.3)

plt.title('Covid19 with top 20 countries')

plt.xlabel('Countries', size=15)

plt.xticks(rotation='45')

plt.ylabel('Number of cases', size=15)

plt.legend()

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='grey')

plt.gca().patch.set_facecolor('0.8')

plt.barh(top20_countries['Country_Region'],top20_countries['Confirmed'], edgecolor='blue', height=0.5)

plt.title('Confirmed cases of top 20 countries', size=15)

plt.xlabel('Confirmed cases', size=15)

plt.ylabel('Countries', size=15)

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='grey')

plt.gca().patch.set_facecolor('0.8')

plt.barh(top20_countries['Country_Region'],top20_countries['Deaths'], edgecolor='b', height=0.5)

plt.title('Death cases of top 20 countries', size=15)

plt.xlabel('Death cases', size=15)

plt.ylabel('Countries', size=15)

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='grey')

plt.gca().patch.set_facecolor('0.8')

plt.barh(top20_countries['Country_Region'],top20_countries['Recovered'], edgecolor='b', height=0.5)

plt.title('Recovered cases of top 20 countries', size=15)

plt.xlabel('Recovered cases', size=15)

plt.ylabel('Countries', size=15)

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='grey')

plt.gca().patch.set_facecolor('0.8')

plt.barh(top20_countries['Country_Region'],top20_countries['Active'], edgecolor='b', height=0.5)

plt.title('Active cases of top 20 countries', size=15)

plt.xlabel('Active cases', size=15)

plt.ylabel('Countries', size=15)

plt.show()
China = temp[temp['Country_Region'] == 'China']

Italy = temp[temp['Country_Region'] == 'Italy']

US = temp[temp['Country_Region'] == 'US']
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.8')

plt.plot(China['Date'], China['Confirmed'], color='red', label='Cases in China')

plt.plot(Italy['Date'], Italy['Confirmed'], color='green', label='Cases in Italy')

plt.plot(US['Date'], US['Confirmed'], color='blue', label='Cases in Unites States')

plt.fill_between(China['Date'], China['Confirmed'], color='orange', alpha=0.3)

plt.fill_between(Italy['Date'], Italy['Confirmed'], color='green', alpha=0.3)

plt.fill_between(US['Date'], US['Confirmed'], color='blue', alpha=0.3)

plt.xlabel('Dates', size=15)

plt.xticks(rotation=90)

plt.ylabel('Confirmed cases')

plt.title('Covid19 spread over time')

plt.legend()

plt.show()
plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.8')

plt.plot(China['Date'], China['Deaths'], color='red', label='Deaths in China')

plt.plot(Italy['Date'], Italy['Deaths'], color='green', label='Deaths in Italy')

plt.plot(US['Date'], US['Deaths'], color='blue', label='Deaths in Unites States')

plt.fill_between(China['Date'], China['Deaths'], color='orange', alpha=0.3)

plt.fill_between(Italy['Date'], Italy['Deaths'], color='green', alpha=0.3)

plt.fill_between(US['Date'], US['Deaths'], color='blue', alpha=0.3)

plt.xlabel('Dates', size=15)

plt.xticks(rotation=90)

plt.ylabel('Count of deaths', size=15)

plt.title('Covid19 Deaths over time', size=15)

plt.legend()

plt.show()
India = temp[temp['Country_Region'] == 'India']



plt.figure(figsize=(14,6))

plt.grid(True, color='w')

plt.gca().patch.set_facecolor('0.8')

plt.plot(India['Date'], India['Confirmed'], color='blue', label='Confirmed cases in India')

plt.plot(India['Date'], India['Recovered'], color='green', label='Recovered cases in India')

plt.plot(India['Date'], India['Deaths'], color='red', label='Death cases in India')

plt.fill_between(India['Date'], India['Confirmed'], color='blue', alpha=0.3)

plt.fill_between(India['Date'], India['Recovered'], color='green', alpha=0.3)

plt.fill_between(India['Date'], India['Deaths'], color='orange', alpha=0.3)

plt.xlabel('Dates', size=15)

plt.xticks(rotation='90')

plt.ylabel('Number of cases', size=15)

plt.title('Covid19 in India over time', size=15)

plt.legend()

plt.show()