import pandas as pd

import numpy as np

import datetime

import requests

import warnings



import matplotlib.pyplot as plt

import matplotlib

import matplotlib.dates as mdates

import seaborn as sns

import squarify

import plotly.offline as py

import plotly_express as px



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import OrdinalEncoder

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima_model import ARIMA

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot



from IPython.display import Image

warnings.filterwarnings('ignore')

%matplotlib inline



age_details = pd.read_csv('../input/covid19-in-pakistan/AgeGroupGenderDetails.csv')

pakistan_covid_19 = pd.read_csv('../input/covid19-in-pakistan/covid_19_pakistan.csv')

hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

#ICMR_details = pd.read_csv('../input/covid19-in-india/ICMRTestingDetails.csv')

ICMR_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')

state_testing = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')

population = pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')



world_population = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')

confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')



pakistan_covid_19['date'] = pd.to_datetime(pakistan_covid_19['date'],dayfirst = True)

state_testing['Date'] = pd.to_datetime(state_testing['Date'])

#ICMR_details['DateTime'] = pd.to_datetime(ICMR_details['DateTime'],dayfirst = True)

#ICMR_details = ICMR_details.dropna(subset=['TotalSamplesTested', 'TotalPositiveCases'])
world_confirmed = confirmed_df[confirmed_df.columns[-1:]].sum()

world_recovered = recovered_df[recovered_df.columns[-1:]].sum()

world_deaths = deaths_df[deaths_df.columns[-1:]].sum()

world_active = world_confirmed - (world_recovered - world_deaths)



labels = ['Active','Recovered','Deceased']

sizes = [world_active,world_recovered,world_deaths]

color= ['#66b3ff','green','red']

explode = []



for i in labels:

    explode.append(0.05)

    

plt.figure(figsize= (15,10))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode,colors = color)

centre_circle = plt.Circle((0,0),0.70,fc='white')



fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title('World COVID-19 Cases',fontsize = 20)

plt.axis('equal')  

plt.tight_layout()

hotspots = ['China','Germany','Iran','Italy','Spain','US','Korea, South','France','Turkey','United Kingdom','Pakistan']

dates = list(confirmed_df.columns[4:])

dates = list(pd.to_datetime(dates))

dates_pakistan = dates[8:]





df1 = confirmed_df.groupby('Country/Region').sum().reset_index()

df2 = deaths_df.groupby('Country/Region').sum().reset_index()

df3 = recovered_df.groupby('Country/Region').sum().reset_index()



global_confirmed = {}

global_deaths = {}

global_recovered = {}

global_active= {}



for country in hotspots:

    k =df1[df1['Country/Region'] == country].loc[:,'1/30/20':]

    global_confirmed[country] = k.values.tolist()[0]



    k =df2[df2['Country/Region'] == country].loc[:,'1/30/20':]

    global_deaths[country] = k.values.tolist()[0]



    k =df3[df3['Country/Region'] == country].loc[:,'1/30/20':]

    global_recovered[country] = k.values.tolist()[0]

    

for country in hotspots:

    k = list(map(int.__sub__, global_confirmed[country], global_deaths[country]))

    global_active[country] = list(map(int.__sub__, k, global_recovered[country]))

    

fig = plt.figure(figsize= (20,20))

plt.suptitle('Active, Recovered, Deaths in Hotspot Countries and Pakistan as of Jun 8',fontsize = 20,y=1.0)

#plt.legend()

k=0

for i in range(1,12):

    ax = fig.add_subplot(6,2,i)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    ax.bar(dates_pakistan,global_active[hotspots[k]],color = 'green',alpha = 0.6,label = 'Active');

    ax.bar(dates_pakistan,global_recovered[hotspots[k]],color='grey',label = 'Recovered');

    ax.bar(dates_pakistan,global_deaths[hotspots[k]],color='red',label = 'Death');   

    plt.title(hotspots[k])

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper left')

    k=k+1



plt.tight_layout(pad=3.0)
hotspots = ['China','Germany','Iran','Italy','Spain','United States','South Korea','France','Turkey','United Kingdom','Pakistan']

country_death_rate = pd.DataFrame(columns = ['country','day1','day2','day3'])

world_population['Population (2020)'] = world_population['Population (2020)']/1000000



d1=[]

d2 =[]

d3 = []

for country in hotspots:

    p = float(world_population[world_population['Country (or dependency)'] == country ]['Population (2020)'])

    if country == 'United States':

        k = global_deaths['US'][-3:]

    elif country == 'South Korea':

        k = global_deaths['Korea, South'][-3:]

    else:

        k = global_deaths[country][-3:]

    d1.append(round(k[0]/p,2))

    d2.append(round(k[1]/p,2))

    d3.append(round(k[2]/p,2))



country_death_rate['country'] = hotspots

country_death_rate['day1'] = d1

country_death_rate['day2'] = d2

country_death_rate['day3'] = d3    



plt.figure(figsize= (10,10))

plt.hlines(y=country_death_rate['country'], xmin=country_death_rate['day1'], xmax=country_death_rate['day3'], color='grey', alpha=0.4);

plt.scatter(country_death_rate['day1'], country_death_rate['country'], color='skyblue', label='13th May');

plt.scatter(country_death_rate['day2'], country_death_rate['country'], color='green', label='14th May');

plt.scatter(country_death_rate['day3'], country_death_rate['country'], color='red', label='15th May');

plt.legend();

plt.title("Death Rate per Million in Hotspot Countries",fontsize=20);

plt.xlabel('Death Rate per Million');

labels = list(age_details['AgeGroup'])

sizes = list(age_details['TotalCases'])



explode = []



for i in labels:

    explode.append(0.05)

    

plt.figure(figsize= (15,10))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode)

centre_circle = plt.Circle((0,0),0.70,fc='white')



fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title('Pakistan - Age Group wise Distribution',fontsize = 20)

plt.axis('equal')  

plt.tight_layout()
labels = ['Male', 'Female']

sizes = []

sizes.append(age_details['Male'].sum())

sizes.append(age_details['Female'].sum())



explode = (0.05 , 0)

colors = ['#66b3ff','#ff9999']



plt.figure(figsize= (15,10))

plt.title('Percentage of Gender',fontsize = 20)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)

plt.axis('equal')

plt.tight_layout()
dates = list(confirmed_df.columns[4:])

dates = list(pd.to_datetime(dates))

dates_pakistan = dates[8:]
# df1 = confirmed_df.groupby('Country/Region').sum().reset_index()

df2 = deaths_df.groupby('Country/Region').sum().reset_index()

df3 = recovered_df.groupby('Country/Region').sum().reset_index()



k = df1[df1['Country/Region']=='Pakistan'].loc[:,'1/30/20':]

pakistan_confirmed = k.values.tolist()[0] 



k = df2[df2['Country/Region']=='Pakistan'].loc[:,'1/30/20':]

pakistan_deaths = k.values.tolist()[0] 



k = df3[df3['Country/Region']=='Pakistan'].loc[:,'1/30/20':]

pakistan_recovered = k.values.tolist()[0] 



plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 11)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Total Confirmed, Active, Death in Pakistan" , fontsize = 20)



ax1 = plt.plot_date(y= pakistan_confirmed,x= dates_pakistan,label = 'Confirmed',linestyle ='-',color = 'b')

ax2 = plt.plot_date(y= pakistan_recovered,x= dates_pakistan,label = 'Recovered',linestyle ='-',color = 'g')

ax3 = plt.plot_date(y= pakistan_deaths,x= dates_pakistan,label = 'Death',linestyle ='-',color = 'r')

plt.legend();
countries = ['Pakistan', 'China','US', 'Italy', 'Spain', 'France']



global_confirmed = []

global_recovered = []

global_deaths = []

global_active = []



for country in countries:

    k =df1[df1['Country/Region'] == country].loc[:,'1/30/20':]

    global_confirmed.append(k.values.tolist()[0]) 



    k =df2[df2['Country/Region'] == country].loc[:,'1/30/20':]

    global_deaths.append(k.values.tolist()[0]) 



    k =df3[df3['Country/Region'] == country].loc[:,'1/30/20':]

    global_deaths.append(k.values.tolist()[0])

plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 11)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Comparison with other Countries" , fontsize = 20)



for i in range(len(countries)):

    plt.plot_date(y= global_confirmed[i],x= dates_pakistan,label = countries[i],linestyle ='-')

plt.legend();
all_state = list(pakistan_covid_19['province'].unique())

#all_state.remove('Unassigned')

#all_state.remove('Nagaland#')

#all_state.remove('Nagaland')

latest = pakistan_covid_19

state_cases = latest.groupby('province')['confirmed'].max().reset_index()

#latest['Active'] = latest['Confirmed'] - (latest['Deaths']- latest['Cured'])

state_cases = state_cases.sort_values('confirmed', ascending= False).fillna(0)

states =list(state_cases['province'][0:7])



states_confirmed = {}

states_deaths = {}

states_recovered = {}

states_active = {}

states_dates = {}



for state in states:

    df = latest[latest['province'] == state].reset_index()

    k = []

   # l = []

   # m = []

   # n = []

    for i in range(1,len(df)):

        k.append(df['confirmed'][i]-df['confirmed'][i-1])

       # l.append(df['Deaths'][i]-df['Deaths'][i-1])

       # m.append(df['Cured'][i]-df['Cured'][i-1])

       # n.append(df['Active'][i]-df['Active'][i-1])

    states_confirmed[state] = k

    #states_deaths[state] = l

    #states_recovered[state] = m

    #states_active[state] = n

    date = list(df['date'])

    states_dates[state] = date[1:]

    

def calc_movingaverage(values ,N):    

    cumsum, moving_aves = [0], [0,0]

    for i, x in enumerate(values, 1):

        cumsum.append(cumsum[i-1] + x)

        if i>=N:

            moving_ave = (cumsum[i] - cumsum[i-N])/N

            moving_aves.append(moving_ave)

    return moving_aves



fig = plt.figure(figsize= (25,17))

plt.suptitle('5-Day Moving Average of Confirmed Cases in Top 15 States',fontsize = 20,y=1.0)

k=0

for i in range(1,8):

    ax = fig.add_subplot(5,3,i)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    ax.bar(states_dates[states[k]],states_confirmed[states[k]],label = 'Day wise Confirmed Cases ') 

    moving_aves = calc_movingaverage(states_confirmed[states[k]],5)

    ax.plot(states_dates[states[k]][:-2],moving_aves,color='red',label = 'Moving Average',linewidth =3)  

    plt.title(states[k],fontsize = 20)

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper left')

    k=k+1

plt.tight_layout(pad=3.0)
def calc_growthRate(values):

    k = []

    for i in range(1,len(values)):

        summ = 0

        for j in range(i):

            summ = summ + values[j]

        rate = (values[i]/summ)*100

        k.append(int(rate))

    return k



fig = plt.figure(figsize= (25,17))

plt.suptitle('Growth Rate in all areas',fontsize = 20,y=1.0)

k=0

for i in range(1,8):

    ax = fig.add_subplot(5,3,i)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    #ax.bar(states_dates[states[k]],states_confirmed[states[k]],label = 'Day wise Confirmed Cases ') 

    growth_rate = calc_growthRate(states_confirmed[states[k]])

    ax.plot_date(states_dates[states[k]][21:],growth_rate[20:],color = '#9370db',label = 'Growth Rate',linewidth =3,linestyle='-')  

    plt.title(states[k],fontsize = 20)

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper left')

    k=k+1

plt.tight_layout(pad=3.0)
df1 = confirmed_df.groupby('Country/Region').sum().reset_index()

df2 = deaths_df.groupby('Country/Region').sum().reset_index()

df3 = recovered_df.groupby('Country/Region').sum().reset_index()



k =df1[df1['Country/Region']=='Pakistan'].loc[:,'2/26/20':]

l =df3[df3['Country/Region']=='Pakistan'].loc[:,'2/26/20':]

pakistan_confirmed = k.values.tolist()[0]



growth_diff = []



for i in range(1,len(pakistan_confirmed)):

    growth_diff.append(pakistan_confirmed[i] / pakistan_confirmed[i-1])



growth_factor = sum(growth_diff)/len(growth_diff)

print('Average growth factor',growth_factor)
prediction_dates = []



start_date = dates_pakistan[len(dates_pakistan) - 1]

for i in range(15):

    date = start_date + datetime.timedelta(days=1)

    prediction_dates.append(date)

    start_date = date

previous_day_cases = global_confirmed[0][len(dates_pakistan) - 1]

predicted_cases = []



for i in range(15):

    predicted_value = previous_day_cases *  growth_factor

    predicted_cases.append(predicted_value)

    previous_day_cases = predicted_value



plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 11)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Predicted Values for the next 15 Days" , fontsize = 20)

ax1 = plt.plot_date(y= predicted_cases,x= prediction_dates,linestyle ='-',color = 'c')
train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
k = df1[df1['Country/Region']=='Pakistan'].loc[:,'1/22/20':]

pakistan_confirmed = k.values.tolist()[0]

data = pd.DataFrame(columns = ['ds','y'])

data['ds'] = dates

data['y'] = pakistan_confirmed



prop=Prophet()

prop.fit(data)

future=prop.make_future_dataframe(periods=30)

prop_forecast=prop.predict(future)

forecast = prop_forecast[['ds','yhat']].tail(30)



fig = plot_plotly(prop, prop_forecast)

fig = prop.plot(prop_forecast,xlabel='Date',ylabel='Confirmed Cases')
arima = ARIMA(data['y'], order=(5, 1, 0))

arima = arima.fit(trend='c', full_output=True, disp=True)

forecast = arima.forecast(steps= 30)

pred = list(forecast[0])

start_date = data['ds'].max()

prediction_dates = []

for i in range(30):

    date = start_date + datetime.timedelta(days=1)

    prediction_dates.append(date)

    start_date = date

plt.figure(figsize= (15,10))

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Predicted Values for the next 15 Days" , fontsize = 20)



plt.plot_date(y= pred,x= prediction_dates,linestyle ='dashed',color = '#ff9999',label = 'Predicted');

plt.plot_date(y=data['y'],x=data['ds'],linestyle = '-',color = 'blue',label = 'Actual');

plt.legend();
train['day'] = train['Date'].dt.day

train['month'] = train['Date'].dt.month

train['dayofweek'] = train['Date'].dt.dayofweek

train['dayofyear'] = train['Date'].dt.dayofyear

train['quarter'] = train['Date'].dt.quarter

train['weekofyear'] = train['Date'].dt.weekofyear

test['day'] = test['Date'].dt.day

test['month'] = test['Date'].dt.month

test['dayofweek'] = test['Date'].dt.dayofweek

test['dayofyear'] = test['Date'].dt.dayofyear

test['quarter'] = test['Date'].dt.quarter

test['weekofyear'] = test['Date'].dt.weekofyear

countries = list(train['Country_Region'].unique())

india_code = countries.index('India')

train = train.drop(['Date','Id'],1)

test =  test.drop(['Date'],1)



train.Province_State.fillna('NaN', inplace=True)

oe = OrdinalEncoder()

train[['Province_State','Country_Region']] = oe.fit_transform(train.loc[:,['Province_State','Country_Region']])



test.Province_State.fillna('NaN', inplace=True)

oe = OrdinalEncoder()

test[['Province_State','Country_Region']] = oe.fit_transform(test.loc[:,['Province_State','Country_Region']])
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','ConfirmedCases','Fatalities']

test_columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State','Country_Region']

train = train[columns]

x = train.drop(['Fatalities','ConfirmedCases'], 1)

y = train['ConfirmedCases']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

test = test[test_columns]

test_india = test[test['Country_Region'] == india_code]
models = []

mse = []

mae = []

rmse = []
lgbm = LGBMRegressor(n_estimators=1300)

lgbm.fit(x_train,y_train)

pred = lgbm.predict(x_test)

lgbm_forecast = lgbm.predict(test_india)

models.append('LGBM')

mse.append(round(mean_squared_error(pred, y_test),2))

mae.append(round(mean_absolute_error(pred, y_test),2))

rmse.append(round(np.sqrt(mean_squared_error(pred, y_test)),2))
rf = RandomForestRegressor(n_estimators=100)

rf.fit(x_train,y_train)

pred = rf.predict(x_test)

rfr_forecast = rf.predict(test_india)

models.append('Random Forest')

mse.append(round(mean_squared_error(pred, y_test),2))

mae.append(round(mean_absolute_error(pred, y_test),2))

rmse.append(round(np.sqrt(mean_squared_error(pred, y_test)),2))
xgb = XGBRegressor(n_estimators=100)

xgb.fit(x_train,y_train)

pred = xgb.predict(x_test)

xgb_forecast = xgb.predict(test_india)

models.append('XGBoost')

mse.append(round(mean_squared_error(pred, y_test),2))

mae.append(round(mean_absolute_error(pred, y_test),2))

rmse.append(round(np.sqrt(mean_squared_error(pred, y_test)),2))