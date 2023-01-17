

import os 

import pandas as pd 

import numpy as np 

import plotly 

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt 

%matplotlib inline 

import warnings 

warnings.filterwarnings('ignore')

import datetime 

import pickle 

from fbprophet import Prophet



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data=pd.read_csv('/kaggle/input/coronavirus-worldwide-cases/time series for covid-19.csv'

,sep=',')

m_data=pd.read_csv('/kaggle/input/moroccocoronavirus/corona_morocco.csv')
# show the head 

data.head()
# filter for Morocco 

maroc= data[data['Country/Region']=='Morocco'].reset_index()

# rename 

maroc.rename(columns={ 'Country/Region':'Country'}, inplace=True)

#drop the Province/State column

maroc.drop(columns='Province/State',axis=1,inplace=True)

#show the df 

maroc.head()
# Start from the first case  appeared in march,02 2020 

maroc= maroc.iloc[40:].reset_index()

maroc.head(10)
# Drop some unuseful columns 

maroc.drop(columns=['level_0','index'],axis=1,inplace=True)

maroc.head()
# calculate active cases 

maroc['Active']=maroc['Confirmed']-(maroc['Recovered']+maroc['Deaths'])

maroc.head()
# group the dataframe by confirmed, recovered, death cases 

confirmed=maroc[['Date','Confirmed']]

deaths=maroc[['Date','Deaths']]

recovered=maroc[['Date','Recovered']]

active=maroc[['Date','Active']]
## Get the daily confirmed cases from the cummulative distrubution of confirmed cases 

#Let's calculate daily cases 

list_confirmed=list(confirmed['Confirmed'])

length=len(list_confirmed)

first_case=list_confirmed[0]

daily_confirmed_cases=[first_case]

for i in range(length):

    if i==0:

        pass

    else:

        diff= list_confirmed[i] - list_confirmed[i-1]

        daily_confirmed_cases.append(diff)

   

#add the list to the dataframe 

maroc['Daily Confirmed']=daily_confirmed_cases
maroc.tail(10)
# we do me for the fatalities to get the daily cases 

list_deaths=list(deaths['Deaths'])

first_case=list_deaths[0]

daily_death_cases=[first_case]

for i in range(len(list_deaths)):

    if i==0:

        pass

    else:

        

        diff= list_deaths[i] - list_deaths[i-1]

        daily_death_cases.append(diff)

#add the list to the dataframe 

maroc['Daily Deaths']=daily_death_cases
# the same for the recovered cases to get recovered daily cases 

list_recovered=list(recovered['Recovered'])

first_case=list_recovered[0]

daily_recovered_cases=[first_case]

for i in range(len(list_recovered)):

    if i==0:

        pass

    else:

        

        diff= list_recovered[i] - list_recovered[i-1]

        daily_recovered_cases.append(diff)

#add the array to the dataframe 

maroc['Daily Recovered Cases']=daily_recovered_cases
maroc.head()
maroc.columns
# rename columns to get rid of the spase 

maroc.rename(columns={ 'Daily Confirmed ':'Daily Confirmed','Daily Death Cases ':'Daily Deaths'}, inplace=True)

daily_cases=maroc[['Date','Daily Confirmed']]

daily_deaths=maroc[['Date','Daily Deaths']]

daily_recovered=maroc[['Date','Daily Recovered Cases']]

daily_cases.head()
daily_deaths.head()
daily_recovered.head()
# show the head the dataframe with Regions 

m_data.head()
m_data.info()
# Empty values everywhere. Let's replace them with 0 instead 

m_data=m_data.fillna(0)

m_data.head()

# now all null values set to zero 
# add the number of Tested cases Column

m_data['Tested']=m_data['Confirmed']+m_data['Excluded']

# add Active Cases Column

m_data['Active']=m_data['Confirmed']-(m_data['Deaths']+m_data['Recovered'])



# show the new dataframe 

m_data.head()
# Rearrange 

m_data=m_data[['Date','Active','Confirmed', 'Deaths', 'Recovered', 'Excluded', 'Tested', 'Beni Mellal-Khenifra', 'Casablanca-Settat', 'Draa-Tafilalet', 'Dakhla-Oued Ed-Dahab', 'Fes-Meknes', 'Guelmim-Oued Noun', 'Laayoune-Sakia El Hamra', 'Marrakesh-Safi', 'Oriental', 'Rabat-Sale-Kenitra', 'Souss-Massa', 'Tanger-Tetouan-Al Hoceima']]

# Then show 

m_data.head()
# Create a list containing names of all Regions 

cols=m_data.columns 

regions=cols[7:]

print(regions)
# Now put them in one single Chart 

df=m_data.copy()

fig = go.Figure()

#Plotting datewise confirmed cases

fig.add_trace(go.Scatter(x=df['Date'], y=df['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Active'], mode='lines+markers', name='Active',line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))

fig.add_trace(go.Scatter(x=df['Date'], y=df['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))

fig.update_layout(title='Coronavirus Cases in Morocco(Cummulative Chart)', xaxis_tickfont_size=14, yaxis=dict(title='Number of Cases'))



fig.show()
# plot the number of tested cases over time 

fig1=go.Figure()

fig1.add_trace(go.Scatter(x=m_data['Date'], y=m_data['Tested'], mode='lines+markers', name='Tested Cases ',line=dict(color='purple', width=2)))

fig1.update_layout(title='Coronairus Tests Over Time ', xaxis_tickfont_size=14, yaxis=dict(title='Number of Cases'))



# till 10/6/2020 more than 350 000 tests were done.
# now let's visualise daily cases 

fig = px.bar(daily_cases, x="Date", y="Daily Confirmed", barmode='group', height=400,color_continuous_midpoint='red')

fig.update_layout(title_text='Coronavirus Daily Cases in Morocco',plot_bgcolor='rgb(230, 230, 230)')



fig.show()

# now let's visualise daily deaths 

fig = px.bar(daily_deaths, x="Date", y="Daily Deaths", barmode='group', height=400,color_continuous_midpoint='red')

fig.update_layout(title_text='Coronavirus Daily Fatalities in Morocco',plot_bgcolor='rgb(230, 230, 230)')

#show 

fig.show()

daily_recovered.head()
# now let's visualise daily deaths 



fig = px.bar(daily_recovered, x='Date', y="Daily Recovered Cases", color='Daily Recovered Cases', orientation='v', height=400,

            title='Recovered Cases in Morocco', color_discrete_sequence = px.colors.cyclical.IceFire)

fig.show()



print("Regions of Morococo :",list(regions))
# create a dataframe to sum all the cases per region 

cases_per_region = m_data[regions].iloc[[-1]].transpose()

cases_per_region.columns=['Total Cases']

cases_per_region
cases_per_region.index
# sort total cases per region 

sorted_cases= cases_per_region.sort_values(by='Total Cases', ascending=True)

plt.figure(figsize=(16,10))

plt.axes(axisbelow=True)

values=sorted_cases.values.reshape(12,)

plt.barh(sorted_cases.index,values,color="blue")

plt.tick_params(size=5,labelsize = 20)

plt.xlabel("# Confirmed Cases",fontsize=18)

plt.title("Total Cases per Region  ",fontsize=20)

plt.grid(alpha=0.3)

for index, value in enumerate(values):

    plt.text(value, index, str(int(value)),fontsize=18,color='red')
# plot the regions data as a pie 

plt.figure(figsize=(20,12))

slices=list(cases_per_region['Total Cases'])

# labels 

Labels =list(cases_per_region.index)

fig = px.pie(cases_per_region, values='Total Cases', names=Labels, title='Total Covid Cases per Region ')

fig.show()



# Convert Date to date time 

confirmed['Date']=pd.to_datetime(confirmed['Date'],format="%Y/%m/%d")

deaths['Date']=pd.to_datetime(deaths['Date'],format="%Y/%m/%d")

recovered['Date']=pd.to_datetime(recovered['Date'],format="%Y/%m/%d")

# change column names to inputs of the prophet model [ds , y ]

confirmed.columns = ['ds','y']

deaths.columns = ['ds','y']

recovered.columns = ['ds','y']



"""  Let's build our prophet model to predict coronavirus new cases 

 a month ahead (30 days ), with an accuracy of 95% """

 

model = Prophet(interval_width=0.95)

#train the model on the present dataframe [confirmed]

model.fit(confirmed)

future_cases = model.make_future_dataframe(periods=30,freq='D',)

# now let's see the new dataframe tail (7 predicted days added to the original one )

future_cases.tail()

#predict the future[yhat] values  with date, and upper and lower limit of y value

forecast_cases = model.predict(future_cases)

forecast_cases[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# plot the future predictions 

model.plot(forecast_cases)
#  plot the weekly distrubution 

model.plot_components(forecast_cases)
# Plot The Real cases from our dataset vs Predicted cases by the model 

plt.figure(figsize=(20,12))

fig = go.Figure()

#Plotting datewise confirmed cases

fig.add_trace(go.Scatter(x=confirmed['ds'], y=confirmed['y'], mode='lines+markers', name='Real Cases',line=dict(color='blue', width=2)))

fig.add_trace(go.Scatter(x=forecast_cases['ds'], y=forecast_cases['yhat'], mode='lines+markers', name='Predicted cases', line=dict(color='green', width=2)))

fig.update_layout(title='Real Covid Cases vs Predicted Cases In Morocco  ', xaxis_tickfont_size=10 ,yaxis=dict(title='Number of Cases'))



fig.show()
# Evaluate our model 

from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(model, initial='60 days', period='30 days', horizon = '30 days')

df_cv.head()
# Plot the Mean Absolute Percentage Error metric 

from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='mape')

fig.show()