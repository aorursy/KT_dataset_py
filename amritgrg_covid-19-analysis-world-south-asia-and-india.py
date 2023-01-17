


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
!pip install pmdarima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy
import datetime
import plotly.offline as py
import plotly_express as px

import pmdarima as pm
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot

from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#world_population = pd.read_csv('population_by_country_2020.csv')
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')


age_details = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
india_covid_19 = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
hospital_beds = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')
individual_details = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')
ICMR_labs = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv')
state_testing = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')
population = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')
df_confirmed = confirmed_df.copy()
df_deaths = deaths_df.copy()
df_recovered = recovered_df.copy() 
df_confirmed['Confirmed'] = df_confirmed.iloc[:,-1]
df_confirmed = df_confirmed.groupby('Country/Region').sum().reset_index()
global_confirmed_cases = df_confirmed.sort_values(['Confirmed'],ascending=True)
global_confirmed_cases.tail(10).plot(x='Country/Region',y='Confirmed', kind='barh', title='Top Countries with Confirmed Cases')
top_countries = global_confirmed_cases.tail(10)
plt.figure(figsize=(12,6))
plt.barh(top_countries['Country/Region'], top_countries['Confirmed'], color='#87479d')
plt.title('Top Countries with Confirmed Cases', fontsize=20)
plt.xlabel('Number of Confirmed Cases',fontsize=15)
plt.ylabel('Contries', fontsize=15)
plt.show()
df_deaths['Deaths'] = df_deaths.iloc[:,-1]
df_deaths = df_deaths.groupby('Country/Region').sum().reset_index()
top_countries = df_deaths.nlargest(10,'Deaths').sort_values(['Deaths'],ascending=True)
#bottom_countries = deaths_df.sort_values('Deaths',ascending=True).head(10)

plt.figure(figsize=(12,8))
plt.barh(top_countries['Country/Region'],top_countries['Deaths'],color='#87479d')
plt.title('Top Countries with Deaths Cases', fontsize=20)
plt.show()
df_recovered['Recovered'] = df_recovered.iloc[:,-1]
df_recovered = df_recovered.groupby('Country/Region').sum().reset_index()
top_countries = df_recovered.nlargest(10,'6/4/20').sort_values(['6/4/20'],ascending=True)

plt.figure(figsize=(12,6))
plt.barh(top_countries['Country/Region'],top_countries['6/4/20'], color='#87479d')
plt.title('Top Countries Revovered Cases',fontsize=20)
plt.ylabel('Countries',fontsize=12)
plt.xlabel('Deaths',fontsize=12)
plt.show()
active_df = pd.DataFrame()

df1 = df_confirmed.groupby('Country/Region').sum().reset_index()  # or simply do df1 = df_confirmed.copy()
df2 = df_deaths.groupby('Country/Region').sum().reset_index()
df3 = df_recovered.groupby('Country/Region').sum().reset_index()

active_df['Country/Region'] = df1['Country/Region']
active_df['Active'] = df1['Confirmed'] - (df3['Recovered'] - df2['Deaths'])

top_active_countries = active_df.nlargest(10,'Active').sort_values(['Active'],ascending=True)

plt.figure(figsize=(12,6))
plt.barh(top_active_countries['Country/Region'],top_active_countries['Active'],color='#87479d')
plt.title('Top Countries Active Cases', fontsize=20)
plt.xlabel('Avtive', fontsize=12)
plt.ylabel('Countries', fontsize=20)
plt.show()
    
world_confirmed = confirmed_df[confirmed_df.columns[-1:]].sum()
world_deaths = deaths_df[deaths_df.columns[-1:]].sum()
world_recovered = recovered_df[recovered_df.columns[-1:]].sum()
world_active = world_confirmed - ( world_recovered - world_deaths)

world_cases = [world_deaths,world_recovered,world_active]
    
plt.figure(figsize=(12,8))
plt.pie(world_cases,labels=['Death','Recovered','Active'], startangle=12, autopct='%1.1f%%',
        colors = ['blue','green','red'])
plt.title('Global Covid-19 Cases')
plt.show()
countries = list(df_confirmed['Country/Region'])
Global_covid = pd.DataFrame(columns=['Country','Confirmed','Deaths','Recovered','Active'])
Global_covid['Country'] = countries
Global_covid['Confirmed'] = df_confirmed['Confirmed']
Global_covid['Deaths'] = df_deaths['Deaths']
Global_covid['Recovered'] = df_recovered['Recovered']
Global_covid['Active'] = active_df['Active']

Global_covid = Global_covid.sort_values(['Confirmed'],ascending=False)
Global_covid = Global_covid.reset_index().drop('index',axis=1)
Global_covid.style.background_gradient(cmap='Reds')

countries = list(Global_covid['Country'].iloc[:10])
dates = confirmed_df.columns[4:]
dates = pd.to_datetime(dates)
dates = list(dates[8:])

confirmed_cases = []
deaths_cases = []
for country in countries:
    m = df_confirmed[df_confirmed['Country/Region']==country].iloc[:,12:]
    confirmed_cases.append(m.values.tolist()[0])
    
    m = df_deaths[df_deaths['Country/Region']==country].iloc[:,12:]
    deaths_cases.append(m.values.tolist()[0])
    
plt.figure(figsize=(12,8))
for i in range(len(countries)):
    plt.plot(dates,confirmed_cases[i],linestyle='-',label=countries[i])

plt.legend();
plt.title("Comparasion in Top 10 most affected Countries\n (Total Confirmed)", fontsize=20)
plt.xlabel('Dates', fontsize=15)
plt.ylabel('Numer of Confirmed Cases', fontsize=15)
plt.xticks(rotation=60)
plt.yticks(fontsize=10)
plt.show()

  
plt.figure(figsize=(12,8))
for i in range(len(countries)):
    plt.plot(dates,deaths_cases[i],linestyle='-',label=countries[i])

plt.legend();
plt.title("Comparasion in Top 10 most affected Countries\n (Total Deaths)", fontsize=20)
plt.xlabel('Dates', fontsize=15)
plt.ylabel('Numer of Deaths Cases', fontsize=15)
plt.xticks(rotation=60)
plt.yticks(fontsize=10)
plt.show()

dates = list(deaths_df.columns[4:]) 
dates = list(pd.to_datetime(dates))
dates = dates[7:]


SAARC_countries = ['Sri Lanka', 'India','Pakistan','Bhutan','Bangladesh','Afghanistan','Maldives','Nepal']

saarc_confirmed = []

for country in SAARC_countries:
    m = df1[df1['Country/Region'] == country].loc[:,'1/30/20':]
    saarc_confirmed.append(m.values.tolist()[0])

plt.figure(figsize=(12,8))
for i in range(len(SAARC_countries)):
    plt.plot(dates,saarc_confirmed[i],linestyle='-', label=SAARC_countries[i])
    
plt.title("Comparasion in SAARC Countries\n ( Total Confirmed)", fontsize=20)
plt.legend();
plt.xlabel('Dates', fontsize=15)
plt.ylabel('Numer of Confirmed Cases', fontsize=15)
plt.xticks(rotation=60)
plt.yticks(fontsize=10)
plt.show()
saarc_deaths = []

for country in SAARC_countries:
    m = df2[df2['Country/Region']==country].loc[:,'1/30/20':]
    saarc_deaths.append(m.values.tolist()[0])
    
plt.figure(figsize=(15,8))
for i in range(len(SAARC_countries)):
    plt.plot(dates,saarc_deaths[i], label=SAARC_countries[i],linestyle='-')
plt.legend();

plt.title("Comparation of SAARC Countries\n (total Deaths)", fontsize=20)
plt.xlabel('Dates', fontsize=15)
plt.ylabel('Number of Deaths', fontsize=15)
plt.yticks(fontsize=10)
plt.xticks(rotation=60)
plt.show()
saarc_recovered = []

for country in SAARC_countries:
    m = df3[df3['Country/Region']==country].loc[:,'1/30/20':]
    saarc_recovered.append(m.values.tolist()[0])
    
plt.figure(figsize=(15,8))
for i in range(len(SAARC_countries)):
    plt.plot_date(x=dates,y=saarc_recovered[i], label=SAARC_countries[i],linestyle='-')
plt.legend();

plt.title("Comparation of SAARC Countries\n ( total Recovered)", fontsize=20)
plt.xlabel('Dates', fontsize=15)
plt.ylabel('Number of Recovered Cases', fontsize=15)
plt.yticks(fontsize=10)
plt.xticks(rotation=60)
plt.show()
saarc = pd.DataFrame()
saarc['Country'] = SAARC_countries
#saarc.columns = ['Confirmed','Deaths','Recovered','Active']
d1 = []
d2 = []
d3 = []
d4 = []

for country in SAARC_countries:
    k = df_confirmed[df_confirmed['Country/Region']==country].iloc[:,-1]
    d1.append(k.values.tolist())
    
    k = df_deaths[df_deaths['Country/Region']==country].iloc[:,-1]
    d2.append(k.values.tolist())
    
    k = df_recovered[df_recovered['Country/Region']==country].iloc[:,-1]
    d3.append(k.values.tolist())
    
    k = active_df[active_df['Country/Region']==country].iloc[:,1]
    d4.append(k.values.tolist())
    

#print(d1)  ----> [[1797], [226713], [85264], [47], [57563], [18054], [1872], [2634]] i.e list of lists
# d2, d3, d4 are also list of lists

# I want to get flat list of above list of lists d1, d2, d3 and d4

flat_d1 = []
for sublist in d1:
    for item in sublist:
        flat_d1.append(item)

# print(flat_d1) ----> [1797, 226713, 85264, 47, 57563, 18054, 1872, 2634] we got flat list of d1

flat_d2 = []
for sublist in d2:
    for item in sublist:
        flat_d2.append(item)
        
flat_d3 = []
for sublist in d3:
    for item in sublist:
        flat_d3.append(item)
        
flat_d4 = []
for sublist in d4:
    for item in sublist:
        flat_d4.append(item)

saarc['Confirmed'] = flat_d1
saarc['Deaths'] = flat_d2
saarc['Recovered'] = flat_d3
saarc['Active'] = flat_d4
saarc['Mortality rate (per 100)'] = np.round(100*saarc['Deaths']/saarc['Confirmed'],2)

saarc = saarc.sort_values(['Confirmed'], ascending=False)
saarc.style.background_gradient(cmap='Reds')
k = len(saarc)
confirmed = saarc.nlargest(k,'Confirmed')
deaths = saarc.nlargest(k,'Deaths')
recovered = saarc.nlargest(k,'Recovered')
active = saarc.nlargest(k,'Active')

plt.figure(figsize=(15,10))
plt.suptitle('Covid-19 Cases in SAARC Countries',fontsize=20)

plt.subplot(221)
plt.title('Confirmed cases')
plt.barh(confirmed['Country'], confirmed['Confirmed'], color =  'pink')

plt.subplot(222)
plt.title('Death Cases')
plt.barh(deaths['Country'], deaths['Deaths'], color='#9370db')

plt.subplot(223)
plt.title('Recovered cases')
plt.barh(recovered['Country'], recovered['Recovered'], color='deeppink')

plt.subplot(224)
plt.title('Active cases')
plt.barh(active['Country'],active['Active'],color= '#9370db')
overall_saarc = pd.DataFrame()
m = saarc.Confirmed.sum()
overall_saarc['Total Confirmed Cases'] = [m]
overall_saarc['Total Deaths'] = saarc.Deaths.sum()
overall_saarc['Total Recovered'] = saarc.Recovered.sum()
overall_saarc['Total Active'] = saarc.Active.sum()
overall_saarc['Mortality Rate (per 100)'] = np.round(100*overall_saarc['Total Deaths']/overall_saarc['Total Confirmed Cases'],2)

overall_saarc.style.background_gradient(cmap='Reds')
t_confirmed = overall_saarc['Total Confirmed Cases'].sum()  # or saarc.Confirmed.sum()
t_deaths = overall_saarc['Total Deaths'].sum()      # or saarc.Deaths.sum()
t_recovered = saarc.Recovered.sum()
t_active = saarc.Active.sum()

cases = [t_deaths,t_recovered,t_active]
labels = ['Deceased', 'Recovered', 'Active']
color= ['#66b3ff','green','red']

explode = []
for i in labels:
    explode.append(0.05)

plt.figure(figsize=(15,10))
plt.pie(cases,labels=labels,autopct='%1.1f%%',startangle=12, colors=color, explode=explode)
circle_center = plt.Circle((0,0),0.60, fc='white')

fig = plt.gcf()
fig.gca().add_artist(circle_center)
plt.title('SAARC Nations COVID-19 Cases',fontsize = 20)
plt.axis('equal')  
plt.tight_layout()
dates = list(deaths_df.columns[4:]) 
dates = list(pd.to_datetime(dates))
dates = dates[7:]

m = df1[df1['Country/Region']=='India'].loc[:,'1/30/20':]
india_confirmed = m.values.tolist()[0]

m = df2[df2['Country/Region']=='India'].loc[:,'1/30/20':]
india_deaths = m.values.tolist()[0]

m = df3[df3['Country/Region']=='India'].loc[:,'1/30/20':]
india_recovered = m.values.tolist()[0]

india_active = list(np.array(india_confirmed) - (np.array(india_recovered) - np.array(india_deaths)))

plt.figure(figsize=(12,8))
plt.plot(dates,india_confirmed,label='Confirmed',linestyle='-', color='blue' )
plt.plot(dates,india_deaths,label='Deaths',linestyle='-',color='green')
plt.plot(dates,india_recovered,label='Recovered',linestyle='-',color='red')
plt.plot(dates,india_active,label='active',linestyle='-',color='black')
plt.legend();

plt.xticks(rotation=90, fontsize=11)
plt.yticks(fontsize = 10)
plt.xlabel('Dates', fontsize=15)
plt.ylabel('Total Cases', fontsize =15)
plt.title('Total Active, Recovered and Death cases in India', fontsize=20)

india_df = pd.DataFrame()
india_df['Dates'] = dates
india_df['TotalConfirm'] = india_confirmed
india_df['TotalDeaths'] = india_deaths
india_df['TotalRecovered'] = india_recovered
india_df['Active'] = india_df['TotalConfirm'] -(india_df['TotalRecovered'] - india_df['TotalDeaths'])


india_df['DailyConfirm'] = india_df['TotalConfirm'].diff()
india_df['DailyDeaths'] = india_df['TotalDeaths'].diff()
india_df['DailyRecovered'] = india_df['TotalRecovered'].diff()
india_df['DailyActive'] = india_df['Active'].diff()
india_df.drop(india_df.index[-1], inplace=True)  #last two rows in dataframe are common so i dropped one of them

plt.figure(figsize=(12,8))
plt.plot(india_df['Dates'],india_df['DailyConfirm'],label='Confirmed', linestyle='-', color='blue')
plt.plot(india_df['Dates'],india_df['DailyDeaths'],label='Deaths', linestyle='-', color='red')
plt.plot(india_df['Dates'],india_df['DailyRecovered'],label='Recovered', linestyle='-', color='green')
#plt.plot(india_df['Dates'],india_df['DailyActive'],label='Active', linestyle='-', color='deeppink')
plt.legend();

plt.title('Daily Covid-19 Cases Trend Analysis in India', fontsize=20)
plt.xticks(rotation=90, fontsize=11)
plt.yticks(fontsize = 10)
plt.xlabel('Dates', fontsize=15)
plt.ylabel('Dailyl Cases', fontsize =15)

print('India Covid Cases in last 10 days')
india_df.tail(10).style.background_gradient(cmap='Reds')
confirmed = india_confirmed[-1] # india_confirmed is a list of confirmed cases in India and -1 will retrive last value of list which is confirmed case in latest date
deaths = india_deaths[-1]
recovered = india_recovered[-1]
active = confirmed - (recovered - deaths)

India_cases = [deaths,recovered,active]
labels = ['Deaths','Recovered','Active']
color = ['red','blue','green']

    
plt.figure(figsize=(12,6))
plt.pie(India_cases,labels=labels,colors=color,autopct='%1.1f%%')
plt.title('India Covid-19 Cases', fontsize=20)
plt.axis('equal')
plt.tight_layout()
states = list(india_covid_19['State/UnionTerritory'].unique())
state_covid = pd.DataFrame(columns=['State','Confirmed','Deaths','Recovered', 'Active'])

c = []
d = []
r = []

for state in states:
    m = india_covid_19[india_covid_19['State/UnionTerritory']==state]['Confirmed'].iloc[-1]
    c.append(m)
    
    m = india_covid_19[india_covid_19['State/UnionTerritory']==state]['Deaths'].iloc[-1]
    d.append(m)
    
    m =  india_covid_19[india_covid_19['State/UnionTerritory']==state]['Cured'].iloc[-1]
    r.append(m)

state_covid['State'] = states
state_covid['Confirmed'] = c
state_covid['Deaths'] = d
state_covid['Recovered'] = r
state_covid['Active'] = state_covid['Confirmed'] - (state_covid['Recovered'] - state_covid['Deaths'])
state_covid['Mortality Rate(per 100)'] = np.round(100*state_covid['Deaths']/state_covid['Confirmed'],2)

state_covid = state_covid.sort_values(['Confirmed'],ascending=False).reset_index().drop('index',axis=1)
state_covid.style.background_gradient(cmap='Greens')
Total_confirmed_india = state_covid['Confirmed'].sum()
Total_deaths_india = state_covid['Deaths'].sum()
print('total Confirmed Cases in india :', Total_confirmed_india)
print('Total deaths in India: ', Total_deaths_india,)
state_covid = state_covid.sort_values(['Confirmed'],ascending=True)
plt.figure(figsize=(12,20))
plt.barh(state_covid['State'],state_covid['Confirmed'])
plt.title('Statewise Covid Cases\n (Confirmed)', fontsize=20)
plt.ylabel('States', fontsize=15)
plt.xlabel('Total Confirmed Cases', fontsize=15)
plt.show()
state_covid = state_covid.sort_values(['Deaths'],ascending=True)
plt.figure(figsize=(15,20))
plt.barh(state_covid['State'],state_covid['Deaths'])
plt.title('Statewise Covid Cases\n (Deaths)', fontsize=20)
plt.ylabel('States', fontsize=15)
plt.xlabel('Total Deaths Cases', fontsize=15)
plt.show()
plt.figure(figsize=(10,8))
plt.title('Death rate per 100 Confirmed cases\n (States wise)', fontsize=20)
plt.ylabel('States', fontsize=15)
plt.xlabel('Death Rates', fontsize=15)
plt.scatter(state_covid['Mortality Rate(per 100)'],state_covid['State'])
plt.show()
states = list(state_testing['State'].unique())
state_covid_test = pd.DataFrame(columns=['State','TotalSamples','TotalNegative','TotalPositive'])

samples = []
negative = []
positive = []

for state in states:
    m = state_testing[state_testing['State']==state]['TotalSamples'].iloc[-1]
    samples.append(m)
    
    m = state_testing[state_testing['State']==state]['Negative'].iloc[-1]
    negative.append(m)
    
    m = state_testing[state_testing['State']==state]['Positive'].iloc[-1]
    positive.append(m)

state_covid_test['State'] = states
state_covid_test['TotalSamples'] = samples
state_covid_test['TotalNegative'] = negative
state_covid_test['TotalPositive'] = positive
state_covid_test['Positive Rate(per 100)'] = np.round(100*state_covid_test['TotalPositive']/state_covid_test['TotalSamples'],2)

state_covid_test.style.background_gradient(cmap='Greens')
age_details
plt.figure(figsize=(10,6))
plt.bar(age_details['AgeGroup'],age_details['TotalCases'])
plt.title('Total Cases in India by Age Group', fontsize=20,)
plt.xlabel('AgeGroup', fontsize=14)
plt.ylabel('total Cases', fontsize=14)
plt.show()
dates = list(deaths_df.columns[4:]) 
dates = list(pd.to_datetime(dates))
dates = dates[7:]

df1 = df_confirmed.groupby('Country/Region').sum().reset_index()
df2 = df_deaths.groupby('Country/Region').sum().reset_index()
df3 = df_recovered.groupby('Country/Region').sum().reset_index()

m = df1[df1['Country/Region']=='Nepal'].loc[:,'1/30/20':]
nepal_confirmed = m.values.tolist()[0]

m = df2[df2['Country/Region']=='Nepal'].loc[:,'1/30/20':]
nepal_deaths = m.values.tolist()[0]

m = df3[df3['Country/Region']=='Nepal'].loc[:,'1/30/20':]
nepal_recovered = m.values.tolist()[0]

plt.figure(figsize=(12,6))
plt.plot(dates,nepal_confirmed,label='Confirmed',linestyle='-', color='blue' )
plt.plot(dates,nepal_deaths,label='Deaths',linestyle='-',color='green')
plt.plot(dates,nepal_recovered,label='Recovered',linestyle='-',color='red')
plt.legend();

plt.xticks(rotation=90, fontsize=11)
plt.yticks(fontsize = 10)
plt.xlabel('Dates', fontsize=20)
plt.ylabel('Total Cases', fontsize =20 )
plt.title('Trends of \n Total Active, Recovered and Death cases in Nepal', fontsize=20)
Nep_confirmed = nepal_confirmed[-1] # nepal_confirmed is a list of confirmed cases in nepal and -1 will retrive last value of list which is confirmed case in latest date
Nep_deaths = nepal_deaths[-1]
Nep_recovered = nepal_recovered[-1]
Nep_active = Nep_confirmed - (Nep_recovered - Nep_deaths)

Nepal_cases = [Nep_deaths,Nep_recovered,Nep_active]
labels = ['Deaths','Recovered','Active']
color = ['red','blue','green']

explode = []
for i in labels:
    explode.append(0.05)
    
plt.figure(figsize=(12,6))
plt.pie(Nepal_cases,labels=labels,colors=color,explode=explode,autopct='%1.1f%%')
circle_centre = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()
fig.gca().add_artist(circle_centre)
plt.title('Nepal Covid-19 Cases', fontsize=20)
plt.axis('equal')
plt.tight_layout()

Nepal_covid_cases = pd.DataFrame()
Nepal_covid_cases['Country'] = ['Nepal']
Nepal_covid_cases['TotalConfirmed'] = Nep_confirmed
Nepal_covid_cases['TotalDeaths'] = Nep_deaths
Nepal_covid_cases['TotalRecovered'] = Nep_recovered
Nepal_covid_cases['TotalActive'] = Nep_active
Nepal_covid_cases.style.background_gradient(cmap='Reds')
Nepal_covid_cases
confirm = df_confirmed.copy()
confirm = confirm.drop('Confirmed', axis=1)
m = confirm[confirm['Country/Region']=='India'].loc[:,'1/22/20':]
india_confirmed = m.values.tolist()[0] 
dates = list(confirm.columns[3:])

data = pd.DataFrame(columns = ['Dates','values'])
data['Dates'] = dates
data['values'] = india_confirmed


def arimamodel(timeseries):
    
    model = pm.auto_arima(timeseries, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=8, max_q=8, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                     
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    return model
    
def plotarima(n_periods,timeseries, model):
    #forecast
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = np.arange(len(timeseries), len(timeseries)+n_periods)
    
    # forecast series and lower and upper confidence bounds
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # plotting
    plt.figure(figsize= (15,10))
    plt.xlabel("Dates",fontsize = 20)
    plt.ylabel('Total cases',fontsize = 20)
    plt.title("Predicted Values for the next 15 Days" , fontsize = 20)
    plt.plot(timeseries)
    plt.plot(fc_series, color='red')
    plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

    plt.legend(("past", "forecast", "95% confidence interval"), loc="upper left")
    plt.show()

automodel = arimamodel(data['values'])
print(automodel.summary())
automodel.plot_diagnostics(figsize=(7,5))
plt.show()
plotarima(15, data['values'], automodel)
m= confirm[confirm['Country/Region']=='India'].iloc[:,3:]
india_confirmed = m.values.tolist()[0] 
ddates = list(confirm.columns[3:])

# creating data frame for prophet model
data = pd.DataFrame(columns = ['ds','y'])
data['ds'] = dates
data['y'] = india_confirmed

prop=Prophet()
prop.fit(data)
future=prop.make_future_dataframe(periods=30)
prop_forecast=prop.predict(future)
forecast = prop_forecast[['ds','yhat']].tail(30)

fig = plot_plotly(prop, prop_forecast)
fig = prop.plot(prop_forecast,xlabel='Date',ylabel='Confirmed Cases')


deaths = df_deaths.copy()
deaths = deaths.drop('Deaths',axis=1)
m = deaths[deaths['Country/Region']=='India'].loc[:,'1/22/20':]
india_deaths = m.values.tolist()[0] 
dates = list(deaths.columns[3:])

data = pd.DataFrame(columns = ['Dates','values'])
data['Dates'] = dates
data['values'] = india_deaths
automodel = arimamodel(data['values'])
print(automodel.summary())
plotarima(15, data['values'], automodel)
m = deaths[deaths['Country/Region']=='India'].loc[:,'1/22/20':]
india_deaths = m.values.tolist()[0] 
dates = list(deaths.columns[3:])

# creating data frame for prophet model
data = pd.DataFrame(columns = ['ds','y'])
data['ds'] = dates
data['y'] = india_confirmed

prop=Prophet()
prop.fit(data)
future=prop.make_future_dataframe(periods=30)
prop_forecast=prop.predict(future)
forecast = prop_forecast[['ds','yhat']].tail(30)

fig = plot_plotly(prop, prop_forecast)
fig = prop.plot(prop_forecast,xlabel='Date',ylabel='Death Cases')







