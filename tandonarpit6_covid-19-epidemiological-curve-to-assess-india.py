import numpy as np

import pandas as pd



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

from plotly.offline import init_notebook_mode, iplot

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots
cleaned_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
cleaned_data.rename(columns={'ObservationDate': 'date', 

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Last Update':'last_updated',

                     'Confirmed': 'confirmed',

                     'Deaths':'deaths',

                     'Recovered':'recovered'

                    }, inplace=True)



# cases 

cases = ['confirmed', 'deaths', 'recovered', 'active']



# Active Case = confirmed - deaths - recovered

cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths'] - cleaned_data['recovered']



# replacing Mainland china with just China

cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')

cleaned_data['country'] = cleaned_data['country'].replace('Korea, South', 'South Korea')



# filling missing values 

cleaned_data[['state']] = cleaned_data[['state']].fillna('')

cleaned_data[cases] = cleaned_data[cases].fillna(0)

cleaned_data.rename(columns={'Date':'date'}, inplace=True)



data = cleaned_data



data['mortality rate'] = (100.* (data['deaths']/data['confirmed'])).round(1)
grouped_china = data[data['country'] == "China"].reset_index()

grouped_china_date = grouped_china.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_china_date = grouped_china_date[grouped_china_date ['confirmed']>= 100]

grouped_china_date['Day']=np.arange(len(grouped_china_date))



grouped_italy = data[data['country'] == "Italy"].reset_index()

grouped_italy_date = grouped_italy.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_italy_date = grouped_italy_date[grouped_italy_date ['confirmed']>= 100]

grouped_italy_date['Day']=np.arange(len(grouped_italy_date))



grouped_spain = data[data['country'] == "Spain"].reset_index()

grouped_spain_date = grouped_spain.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_spain_date = grouped_spain_date[grouped_spain_date ['confirmed']>= 100]

grouped_spain_date['Day']=np.arange(len(grouped_spain_date))



grouped_uk = data[data['country'] == "United Kingdom"].reset_index()

grouped_uk_date = grouped_uk.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_uk_date = grouped_uk_date[grouped_uk_date ['confirmed']>= 100]

grouped_uk_date['Day']=np.arange(len(grouped_uk_date))



grouped_us = data[data['country'] == "US"].reset_index()

grouped_us_date = grouped_us.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_us_date = grouped_us_date[grouped_us_date ['confirmed']>= 100]

grouped_us_date['Day']=np.arange(len(grouped_us_date))



grouped_india = data[data['country'] == "India"].reset_index()

grouped_india_date = grouped_india.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_india_date = grouped_india_date[grouped_india_date ['confirmed']>= 100]

grouped_india_date['Day']=np.arange(len(grouped_india_date))



grouped_france = data[data['country'] == "France"].reset_index()

grouped_france_date = grouped_france.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_france_date = grouped_france_date[grouped_france_date ['confirmed']>= 100]

grouped_france_date['Day']=np.arange(len(grouped_france_date))



grouped_germany = data[data['country'] == "Germany"].reset_index()

grouped_germany_date = grouped_germany.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_germany_date = grouped_germany_date[grouped_germany_date ['confirmed']>= 100]

grouped_germany_date['Day']=np.arange(len(grouped_germany_date))



grouped_brazil = data[data['country'] == "Brazil"].reset_index()

grouped_brazil_date = grouped_brazil.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_brazil_date = grouped_brazil_date[grouped_brazil_date ['confirmed']>= 100]

grouped_brazil_date['Day']=np.arange(len(grouped_brazil_date))



grouped_russia = data[data['country'] == "Russia"].reset_index()

grouped_russia_date = grouped_russia.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_russia_date = grouped_russia_date[grouped_russia_date ['confirmed']>= 100]

grouped_russia_date['Day']=np.arange(len(grouped_russia_date))
grouped_china_date['mortality_rate']= (100. * (grouped_china_date['deaths']/ grouped_china_date['confirmed'])).round(1)

grouped_italy_date['mortality_rate']= (100. * (grouped_italy_date['deaths']/ grouped_italy_date['confirmed'])).round(1)

grouped_spain_date['mortality_rate']= (100. * (grouped_spain_date['deaths']/ grouped_spain_date['confirmed'])).round(1)

grouped_uk_date['mortality_rate']= (100. * (grouped_uk_date['deaths']/ grouped_uk_date['confirmed'])).round(1)

grouped_us_date['mortality_rate']= (100. * (grouped_us_date['deaths']/ grouped_us_date['confirmed'])).round(1)

grouped_india_date['mortality_rate']= (100. * (grouped_india_date['deaths']/ grouped_india_date['confirmed'])).round(1)

grouped_france_date['mortality_rate']= (100. * (grouped_france_date['deaths']/ grouped_france_date['confirmed'])).round(1)

grouped_germany_date['mortality_rate']= (100. * (grouped_germany_date['deaths']/ grouped_germany_date['confirmed'])).round(1)

grouped_brazil_date['mortality_rate']= (100. * (grouped_brazil_date['deaths']/ grouped_brazil_date['confirmed'])).round(1)

grouped_russia_date['mortality_rate']= (100. * (grouped_russia_date['deaths']/ grouped_russia_date['confirmed'])).round(1)
total_days = (data['date'].max() - data['date'].min()).days

temp =100



theoretical_3 = [temp]

days_3= []



for x in range(0,total_days,3):

        days_3.append(x)

        temp = temp * 2

        theoretical_3.append(temp)
temp =100



theoretical_7 = [temp]

days_7= []



for x in range(0,total_days,7):

        days_7.append(x)

        temp = temp * 2

        theoretical_7.append(temp)
print("External Data")

print(f"Earliest Entry: {data['date'].min()}")

print(f"Last Entry:     {data['date'].max()}")

print(f"Total Days:     {data['date'].max() - data['date'].min()}")
fig = go.Figure()



fig.add_trace(go.Scatter(x= grouped_china_date.Day, y= grouped_china_date.confirmed, name ='China',

                         line=dict(color='firebrick', width=4)))



fig.add_trace(go.Scatter(x= grouped_russia_date.Day, y= grouped_russia_date.confirmed, name ='Russia',

                         line=dict(color='royalblue', width=4)))



fig.add_trace(go.Scatter(x= grouped_spain_date.Day, y= grouped_spain_date.confirmed, name='Spain',

                         line=dict(color='yellow', width=4)))



fig.add_trace(go.Scatter(x= grouped_uk_date.Day, y= grouped_uk_date.confirmed, name='UK',

                         line=dict(color='goldenrod', width=4)))



fig.add_trace(go.Scatter(x= grouped_brazil_date.Day, y= grouped_brazil_date.confirmed, name='Brazil',

                         line=dict(color='pink', width=4)))





fig.add_trace(go.Scatter(x= grouped_us_date.Day, y= grouped_us_date.confirmed, name='US',

                         line=dict(color='green', width=4)))



fig.add_trace(go.Scatter(x= grouped_india_date.Day, y= grouped_india_date.confirmed, name='India',

                         line=dict(color='orange', width=4)))



#fig.add_trace(go.Scatter(x= days_3, y= theoretical_3, name='cases doubling every 3 days',

 #                        line=dict(color='white', width=4, dash= 'dot')))



#fig.add_trace(go.Scatter(x= days_7, y= theoretical_7, name='cases doubling every 7 days',

 #                        line=dict(color='white', width=4, dash= 'dot')))



fig.update_layout(title='Curve comparing COVID-19 in different countries',

                   xaxis_title='Day',

                   yaxis_title='Confirmed Cases')



fig.update_yaxes(type="log")



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x= grouped_china_date.Day, y= grouped_china_date.mortality_rate, name ='China',

                         line=dict(color='firebrick', width=4)))



fig.add_trace(go.Scatter(x= grouped_italy_date.Day, y= grouped_italy_date.mortality_rate, name ='Italy',

                         line=dict(color='royalblue', width=4)))



fig.add_trace(go.Scatter(x= grouped_spain_date.Day, y= grouped_spain_date.mortality_rate, name='Spain',

                         line=dict(color='yellow', width=4)))



fig.add_trace(go.Scatter(x= grouped_germany_date.Day, y= grouped_germany_date.mortality_rate, name='Germany',

                         line=dict(color='purple', width=4)))



fig.add_trace(go.Scatter(x= grouped_uk_date.Day, y= grouped_uk_date.mortality_rate, name='UK',

                         line=dict(color='goldenrod', width=4)))



fig.add_trace(go.Scatter(x= grouped_france_date.Day, y= grouped_france_date.mortality_rate, name='France',

                         line=dict(color='pink', width=4)))



fig.add_trace(go.Scatter(x= grouped_us_date.Day, y= grouped_us_date.mortality_rate, name='US',

                         line=dict(color='green', width=4)))



fig.add_trace(go.Scatter(x= grouped_india_date.Day, y= grouped_india_date.mortality_rate, name='India',

                         line=dict(color='orange', width=4)))



fig.update_layout(title='Curve comparing mortality rates across different countries',

                   xaxis_title='Day',

                   yaxis_title='Mortality rates')



#fig.update_yaxes(type="log")



fig.show()
mortality_data = data[data['date']== data['date'].max()]

mortality_data = mortality_data.groupby('country')['confirmed','deaths'].sum().reset_index()

mortality_data ['mortality rate']= (100. * (mortality_data['deaths']/ mortality_data['confirmed'])).round(1)

mortality_data = mortality_data.sort_values(by=['confirmed'], ascending = False)
fig = px.bar(mortality_data.head(10), x='country', y ='mortality rate', height= 500, text = 'mortality rate')

fig.update_traces(textposition = 'outside')

fig.show()