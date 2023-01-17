import numpy as np 

import pandas as pd 

import datetime

import requests

import warnings

import random

import squarify

import matplotlib

import seaborn as sns

import matplotlib as mpl

import plotly.offline as py

import plotly_express as px

from sklearn.svm import SVR

import statsmodels.api as sm

from functools import partial

from fbprophet import Prophet

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from lightgbm import LGBMRegressor

from scipy.optimize import minimize

from sklearn.pipeline import Pipeline

from statsmodels.tsa.arima_model import ARIMA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic



from IPython.display import Image

warnings.filterwarnings('ignore')

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Weights -> In the previous weeks (wk 2/3/4) of the competition phase weight was assigned according to the population of the region specified



confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')



world_confirmed = confirmed_df[confirmed_df.columns[-1:]].sum()

world_recovered = recovered_df[recovered_df.columns[-1:]].sum()

world_deaths = deaths_df[deaths_df.columns[-1:]].sum()

world_active = world_confirmed - (world_recovered - world_deaths)



labels = ['Active','Recovered','Deceased']

sizes = [world_active,world_recovered,world_deaths]

color= ['red','green','black']

explode = []



for i in labels:

    explode.append(0.05)



plt.figure(figsize= (15,10))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=8, explode =explode,colors = color)

centre_circle = plt.Circle((0,0),0.70,fc='white')



fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title('World COVID-19 Cases',fontsize = 20)

plt.axis('equal')  

plt.tight_layout()
## DATA READING



df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")

df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')



## PRE-PROCESSING



df_confirmed = df_confirmed.rename(columns={"Province/State":"state","Country/Region": "country"})

df_covid19 = df_covid19.drop(["People_Tested","People_Hospitalized","UID","ISO3","Mortality_Rate"],axis =1)

df_deaths = df_deaths.rename(columns={"Province/State":"state","Country/Region": "country"})

df_covid19 = df_covid19.rename(columns={"Country_Region": "country"})

df_covid19["Active"] = df_covid19["Confirmed"]-df_covid19["Recovered"]-df_covid19["Deaths"]



# Changing the conuntry names as required by pycountry_convert Lib

df_deaths.loc[df_deaths['country'] == "US", "country"] = "USA"

df_deaths.loc[df_deaths['country'] == 'Korea, South', "country"] = 'South Korea'

df_deaths.loc[df_deaths['country'] == 'Taiwan*', "country"] = 'Taiwan'

df_deaths.loc[df_deaths['country'] == 'Congo (Kinshasa)', "country"] = 'Democratic Republic of the Congo'

df_deaths.loc[df_deaths['country'] == "Cote d'Ivoire", "country"] = "Côte d'Ivoire"

df_deaths.loc[df_deaths['country'] == "Reunion", "country"] = "Réunion"

df_deaths.loc[df_deaths['country'] == 'Congo (Brazzaville)', "country"] = 'Republic of the Congo'

df_deaths.loc[df_deaths['country'] == 'Bahamas, The', "country"] = 'Bahamas'

df_deaths.loc[df_deaths['country'] == 'Gambia, The', "country"] = 'Gambia'



countries = np.asarray(df_confirmed["country"])

countries1 = np.asarray(df_covid19["country"])

# Continent_code to Continent_names

continents = {

    'NA': 'North America',

    'SA': 'South America', 

    'AS': 'Asia',

    'OC': 'Australia',

    'AF': 'Africa',

    'EU' : 'Europe',

    'na' : 'Others'}





# Defininng Function for getting continent code for country.

def country_to_continent_code(country):

    try:

        return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))

    except :

        return 'na'



#Collecting Continent Information

df_deaths.insert(2,"continent",  [continents[country_to_continent_code(country)] for country in countries[:]])

df_covid19.insert(1,"continent",  [continents[country_to_continent_code(country)] for country in countries1[:]])



df_deaths[df_deaths["continent" ]== 'Others']

df_deaths = df_deaths.replace(np.nan, '', regex=True)



df_countries_cases = df_covid19.copy().drop(['Lat','Long_','continent','Last_Update'],axis =1)

df_countries_cases.index = df_countries_cases["country"]

df_countries_cases = df_countries_cases.drop(['country'],axis=1)



df_countries_cases.fillna(0,inplace=True)



## VISUALIZATION



temp_df = pd.DataFrame(df_countries_cases['Deaths'])

temp_df = temp_df.reset_index()

fig = px.choropleth(temp_df, locations="country",

                    color=np.log10(temp_df["Deaths"]+1), 

                    hover_name="country", 

                    hover_data=["Deaths"],

                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")

fig.update_geos(fitbounds="locations", visible=False)

fig.update_coloraxes(colorbar_title="Deaths (Log Scale)",colorscale="Reds")



fig.show()
f = plt.figure(figsize=(10,5))

f.add_subplot(111)



plt.axes(axisbelow=True)

plt.barh(df_countries_cases.sort_values('Deaths')["Deaths"].index[-10:],df_countries_cases.sort_values('Deaths')["Deaths"].values[-10:],color="crimson")

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Deaths Cases",fontsize=18)

plt.title("Top 10 Countries (Deaths Cases)",fontsize=20)

plt.grid(alpha=0.3,which='both')
hotspots = ['China','Germany','Iran','Italy','Spain','US','Korea, South','France','Turkey','United Kingdom','India']

dates = list(confirmed_df.columns[4:])

dates = list(pd.to_datetime(dates))

dates_india = dates[8:]



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

    

fig = plt.figure(figsize= (15,25))

plt.suptitle('Active, Recovered, Deaths in Hotspot Countries and India as of May 15',fontsize = 13,y=1.0)

#plt.legend()

k=0

for i in range(1,12):

    ax = fig.add_subplot(6,2,i)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    ax.bar(dates_india,global_active[hotspots[k]],color = 'red',alpha = 0.6,label = 'Active');

    ax.bar(dates_india,global_recovered[hotspots[k]],color='green',label = 'Recovered');

    ax.bar(dates_india,global_deaths[hotspots[k]],color='black',label = 'Death');   

    plt.title(hotspots[k])

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper left')

    k=k+1



plt.tight_layout(pad=3.0)
us = latest_data.loc[latest_data['Country_Region'] == 'US']

us.drop('Admin2', axis=1, inplace=True)



from urllib.request import urlopen

import json

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)



us_min = us["Confirmed"].min()

us_mean = us["Confirmed"].mean()

us_max = us["Confirmed"].max()

us_med = us["Confirmed"].median()



fig = px.choropleth_mapbox(us, geojson=counties, locations="FIPS", color='Confirmed',

                           hover_name="Province_State",

                           color_continuous_scale="OrRd",

                           range_color=(us_med,us_mean),

                           mapbox_style="carto-positron",

                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},

                           opacity=0.4,

                           labels={'Confirmed':'Confirmed Case Number'}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
cluster_data = pd.read_csv("../input/covid19-useful-features-by-country/Countries_usefulFeatures.csv")

age_df = cluster_data[["Country_Region","Mean_Age"]]

sns.distplot(a=age_df['Mean_Age'], kde=False)
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")



train_copy = train.copy()

test_copy = test.copy()



train['day']=pd.to_datetime(train.Date,format='%Y-%m-%d').dt.day

train['month']=pd.to_datetime(train.Date,format='%Y-%m-%d').dt.month



test['day']=pd.to_datetime(test.Date,format='%Y-%m-%d').dt.day

test['month']=pd.to_datetime(test.Date,format='%Y-%m-%d').dt.month



train.columns = map(str.lower, train.columns)

train = train.rename(columns = {'county': 'country', 'province_state': 'state', 'country_region': 'region', 'target': 'case', 'targetvalue':'case_value'}, inplace = False)



tc_data = pd.read_csv("../input/number-of-cases-in-the-city-covid19-turkey/number_of_cases_in_the_city.csv")
tc_list = list(range(1, 82))

tc_data.insert(0, "id", tc_list, True) 



import plotly.express as px



more_case = tc_data.sort_values(by='Number of Case', ascending=False)



fig = px.pie(

    more_case.head(5),

    values = "Number of Case",

    names = "Province",

    color_discrete_sequence = px.colors.sequential.RdBu)



fig.update_traces(textposition="inside", textinfo="percent+label")

fig.show()
import plotly.express as px



# loading Turkey's geoplot json file

from urllib.request import urlopen

import json

with open("../input/geoplot/tr-cities-utf8.json") as f:

    cities = json.load(f)



mini = tc_data["Number of Case"].min()

average = tc_data["Number of Case"].mean()

#tc_data.drop('id', axis=1, inplace=True)

    

fig = px.choropleth_mapbox(tc_data, geojson=cities, locations=tc_data.id, color=(tc_data["Number of Case"]),

                           hover_name="Province",

                           range_color= (mini,average),

                           color_continuous_scale='amp',

                           mapbox_style="carto-positron",

                           zoom=4, opacity=0.7,center = {"lat": 38.963745, "lon": 35.243322},

                           labels={'color':'Number of Case'})



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
less_case = tc_data.sort_values(by='Number of Case', ascending=True)



fig = px.bar(

    less_case.head(10),

    x = "Province",

    y = "Number of Case")

fig.update_layout(barmode="group")

fig.update_traces(marker_color='rosybrown')

fig.show()
import matplotlib as mpl



tc = train.loc[train.region == 'Turkey']



tc.drop('country', axis=1, inplace=True)

tc.drop('state', axis=1, inplace=True)

tc.drop('region', axis=1, inplace=True)

tc.drop('population', axis=1, inplace=True)



tc_1=tc['case_value'].groupby(tc['case']).sum()



fatal_tc=tc[tc['case']=='Fatalities']

conf_tc=tc[tc['case']=='ConfirmedCases']



labels =[tc_1.index[0],tc_1.index[1]]

sizes = [tc_1[0],tc_1[1]]

explode = (0, 0.08)  

plt.figure(figsize = (8,8))



plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',textprops={'fontsize': 14},startangle=90)

plt.show()
px.line(data_frame=conf_tc, x="date", y='case_value',hover_name="case")
fig = px.line(data_frame=fatal_tc, x="date", y='case_value',hover_name="case", color_discrete_map={'case_value': 'red'})

fig.show()
with plt.style.context('fivethirtyeight'):

    dategroup=tc.groupby('month').mean()

    fig, ax = plt.subplots(figsize=(20,6))

    ax.xaxis.set(ticks=range(0,13)) # Manually set x-ticks

    dategroup['case_value'].plot(x=tc.month)
from pandas import Series

from math import sqrt



# metrics

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm



# forecasting model

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from statsmodels.tsa.arima_model import ARIMA



# for analysis

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from shapely.geometry import LineString



import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

import seaborn as sns

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 7



from IPython.display import display, HTML

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train_original=pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test_original=pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

train_original.sample(3)
# Train Data Cleaning

train_original=train_original.drop(["County"], axis=1)

train_original=train_original.drop(["Province_State"], axis=1)

train_original=train_original.drop(["Population"], axis=1)

train_original=train_original.drop(["Weight"], axis=1)

train_original=train_original.drop(["Id"], axis=1)



train_original = pd.DataFrame(train_original[(train_original['Country_Region'] == 'Turkey') & (train_original['Target'] == 'ConfirmedCases')])

train_original=train_original.drop(["Country_Region"], axis=1)

train_original=train_original.drop(["Target"], axis=1)



# Test Data Cleaning

test_original=test_original.drop(["County"], axis=1)

test_original=test_original.drop(["Province_State"], axis=1)

test_original=test_original.drop(["Population"], axis=1)

test_original=test_original.drop(["Weight"], axis=1)



test_original = pd.DataFrame(test_original[(test_original['Country_Region'] == 'Turkey') & (test_original['Target'] == 'ConfirmedCases')])

test_original=test_original.drop(["Country_Region"], axis=1)

test_original=test_original.drop(["Target"], axis=1)



test_original.dropna(inplace=True)

test_original.dropna(inplace=True)

test_original.drop(test_original.tail(1).index, inplace=True)



train_df=train_original.copy()

test_df=test_original.copy()



train_original['Date']=pd.to_datetime(train_original.Date, format='%Y/%m/%d')

test_original['Date']=pd.to_datetime(test_original.Date, format='%Y/%m/%d')

train_df['Date']=pd.to_datetime(train_df.Date, format='%Y/%m/%d')

test_df['Date']=pd.to_datetime(test_df.Date, format='%Y/%m/%d')



# generate day, month, year feature

for i in (train_original, test_original, train_df, test_df):

    i['year']=i.Date.dt.year

    i['month']=i.Date.dt.month

    i['day']=i.Date.dt.day

    i['hour']=i.Date.dt.hour

    

# sampling for daily basis

train_df.index=train_df.Date

test_df.index=test_df.Date



train_df=train_df.resample('D').mean()

test_df=test_df.resample('D').mean()
train_df.head(3) #Last Version
train=train_df.loc['2020-04-11':'2020-05-16']

valid=train_df.loc['2020-05-17':'2020-07-10']

plt.figure(figsize=(20,7))



train.TargetValue.plot(label='Train data')

valid.TargetValue.plot(label='Valid data')

plt.legend(loc='best')
rolmean=train.TargetValue.rolling(window=7).mean() #for 7 days -> roll.mean: pencere gezdirip ortalama alma

rolstd=train.TargetValue.rolling(window=7).std()

rolmean.dropna(inplace=True)

rolstd.dropna(inplace=True)



plt.figure(figsize=(17,7))

rolmean.plot(label='Rolmean', color='green')

rolstd.plot(label='rolstd')

train.TargetValue.plot(label='Train')

plt.legend(loc='best')
dftest=adfuller(train.TargetValue, autolag='AIC')

dfout=pd.Series(dftest[0:4], index=['Test statistics', 'p-value', '#Lags used', 'Number of observation used'])

for key, val in dftest[4].items():

    dfout['Critical value (%s)'%key]=val



print(dfout)
# estimating trend

train_count_log=np.log(train.TargetValue)



# make TS to be stationary

moving_avg=train_count_log.rolling(window=7).mean()

moving_std=train_count_log.rolling(window=7).std()

plt.figure(figsize=(17,7))



train_count_log.plot(label='Log Scale')

moving_avg.plot(label='moving_avg')

moving_std.plot(label='moving_std')



plt.legend(loc='best')
# Varyasyonun yüksek olduğu yer.

dif_log=train_count_log-moving_avg

dif_log.dropna(inplace=True)

dif_log.plot()
def test_stationary(timeseries):

    # determine roling stats

    mov_avg=timeseries.rolling(window=7).mean()

    mov_std=timeseries.rolling(window=7).std()

    #plot rolling stats

    plt.figure(figsize=(12,7))

    timeseries.plot(label='Original')

    mov_avg.plot(label='Mov avg')

    mov_std.plot(label='Mov std')

    plt.legend(loc='best')

    plt.title('Rolling mean & standard deviation')

    

    # dickey-fuller test

    print('Result of Dickey-fuller test')

    dftest=adfuller(timeseries, autolag='AIC')

    dfout=pd.Series(dftest[:4], index=['Test stats', 'p-value', '#Lag used', 'Number of observation used'])

    for key, val in dftest[4].items():

        dfout['Critical value (%s)'%key]=val

    print(dfout)
test_stationary(dif_log)
plt.figure(figsize=(12,7))

edw_avg=train_count_log.ewm(halflife=7, min_periods=0, adjust=True).mean()

train_count_log.plot(label='Log scale')

edw_avg.plot(label='Exponential Decay Weight MA')
dif_edw=train_count_log-edw_avg

dif_edw = dif_edw.replace([np.inf, -np.inf], np.nan)

dif_edw.dropna(inplace=True)

test_stationary(dif_edw)
dif_shift=train_count_log-train_count_log.shift()

dif_shift = dif_shift.replace([np.inf, -np.inf], np.nan)

dif_shift.dropna(inplace=True)

test_stationary(dif_shift)
decom=seasonal_decompose(dif_edw, freq=3)



trend=decom.trend

seasonal=decom.seasonal

residual=decom.resid



fig=plt.figure(figsize=(15,8))

plt.subplot(211)

train_count_log.plot(label='Original')

plt.title("Original")

plt.subplot(212)

trend.plot(label='Trend')

plt.title("Trend")



'''

plt.subplot(413)

seasonal.plot(label='Seasonal')

plt.title("Seasonal")

plt.subplot(414)

residual.plot(label='Residual')

plt.title("Residual")

fig.tight_layout()

'''



decom_log_data=residual

decom_log_data = decom_log_data.replace([np.inf, -np.inf], np.nan)

decom_log_data.dropna(inplace=True)

#test_stationary(decom_log_data)
## AR MODEL:



train_count_log = train_count_log.replace([np.inf, -np.inf], np.nan)

train_count_log.dropna(inplace=True)

model=ARIMA(train_count_log, order=(4,1,0))

results_AR=model.fit(disp=0)



'''

The Residual sum of Squares (RSS) is defined as below and is used in the Least Square Method 

in order to estimate the regression coefficient.

The smallest residual sum of squares is equivalent to the largest r squared.

The deviance calculation is a generalization of residual sum of squares.

Squared loss = (y−y^)2

'''



plt.figure(figsize=(18,6))

dif_edw.plot(label='Exponentian Decay Differentiation')

results_AR.fittedvalues.dropna(inplace=True)

results_AR.fittedvalues.plot(label='Results AR')

df=pd.concat([results_AR.fittedvalues, dif_edw], axis=1).dropna()

plt.title('AR MODEL /RSS: %.4f'%sum((df[0]-df['TargetValue'])**2))
## MA MODEL

model=ARIMA(train_count_log, order=(2,1,1))

results_MA=model.fit(disp=0)



plt.figure(figsize=(18,6))

dif_edw.plot(label='Exponentian Decay Differentiation')

results_MA.fittedvalues.dropna(inplace=True)

results_MA.fittedvalues.plot(label='Results AR')

df=pd.concat([results_MA.fittedvalues, dif_edw], axis=1).dropna()

plt.title('MA MODEL /RSS: %.4f'%sum((df[0]-df['TargetValue'])**2))
## ARIMA MODEL

model=ARIMA(train_count_log, order=(4,1,2))

results_ARIMA=model.fit(disp=0)



plt.figure(figsize=(18,6))

dif_edw.plot(label='Exponentian Decay Differentiation')

results_ARIMA.fittedvalues.dropna(inplace=True)

results_ARIMA.fittedvalues.plot(label='Results AR')

df=pd.concat([results_ARIMA.fittedvalues, dif_edw], axis=1).dropna()

plt.title('ARIMA MODEL /RSS: %.4f'%sum((df[0]-df['TargetValue'])**2))
# using AR model

pred_ar_dif=pd.Series(results_AR.fittedvalues, copy=True)

pred_ar_dif_cumsum=pred_ar_dif.cumsum()



pred_ar_log=pd.Series(train_count_log.iloc[0], index=train_count_log.index)

pred_ar_log=pred_ar_log.add(pred_ar_dif_cumsum, fill_value=0)

pred_ar_log.head()



# inverse of log is exp

pred_ar=np.exp(pred_ar_log)

plt.figure(figsize=(12,7))

train.TargetValue.plot(label='Train')

pred_ar.plot(label='Pred')
def validation(order):

    # forecasting for validation

    valid_count_log=list(np.log(valid.TargetValue).values)

    history = list(train_count_log.values)

    model = ARIMA(history, order=order)

    model_fit = model.fit(disp=0)

    output = model_fit.forecast(steps=len(valid))

    mse = mean_squared_error(valid_count_log, output[0])

    rmse = np.sqrt(mse)

    print('Test MSE: %.3f' % mse)

    print('Test RMSE: %.3f' % rmse)

    

    fig=plt.figure(figsize=(12,7))

    # reverse transform

    pred=np.exp(output[0])

    pred=pd.Series(pred, index=valid.index)

    valid.TargetValue.plot(label='Valid')

    pred.plot(label='Pred')

    plt.legend(loc='best')

    

    fig=plt.figure(figsize=(18,7))

    train.TargetValue.plot(label='Train')

    valid.TargetValue.plot(label='Valid')

    pred.plot(label='Pred', color='black')
validation((2,1,2))
'''def arima_predict_hourly(data, arima_order):

    # forecasting for testing (Daily based forecasting)

    

    history = data

    model = ARIMA(history, order=arima_order)

    model_fit = model.fit(disp=0)

    output = model_fit.forecast(steps=len(test_original))



    submit=test_original.copy()

    submit.index=submit.ID

    submit['Count']=np.exp(output[0])

    submit.drop(['Unnamed: 0','ID','Datetime','year','month','day','hour'], axis=1, inplace=True)

    

    # plot result

    plt.figure(figsize=(12,7))

    train_original.index=train_original.Datetime

    submit.index=test_original.Datetime



    train_original.TargetValue.plot(label='Train')

    submit.TargetValue.plot(label='Pred')

    return submit'''



from pandas import DataFrame

# forecasting for testing (daily based forecasting)



h = list(np.log(train_original.TargetValue).values)

history = DataFrame(h,columns=['values'])

history = history.replace([np.inf, -np.inf], np.nan)

history.fillna(0, inplace=True)



model = ARIMA(history, order=(2,0,1))

model_fit = model.fit(disp=0)

output = model_fit.forecast(steps=len(test_original))



submit=test_original.copy()

submit.index=submit.ForecastId

submit['TargetValue']=np.exp(output[0])

submit.drop(['ForecastId','Date','year','month','day','hour'], axis=1, inplace=True)
# plot result

plt.figure(figsize=(18,7))

train_original.index=train_original.Date

submit.index=test_original.Date



train_original.TargetValue.plot(label='Train')

submit.TargetValue.plot(label='Pred')
# evaluate an ARIMA model for a given order (p,d,q)

def evaluate_arima_model(arima_order):

    # forecasting for validation

    valid_count_log=list(np.log(valid.TargetValue).values)

    history = list(train_count_log.values)

    model = ARIMA(history, order=arima_order)

    model_fit = model.fit(disp=0)

    output = model_fit.forecast(steps=len(valid))

    mse = mean_squared_error(valid_count_log, output[0])

    rmse = np.sqrt(mse)

#     print('Test MSE: %.3f' % mse)

#     print('Test RMSE: %.3f' % rmse)

    return mse





# evaluate combinations of p, d and q values for an ARIMA model

def evaluate_models(p_values, d_values, q_values):

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    mse = evaluate_arima_model(order)

                    if mse < best_score:

                        best_score, best_cfg = mse, order

                    print('ARIMA%s MSE=%.3f' % (order,mse))

                except:

                    continue

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
# evaluate parameters

p_values = [0, 1, 2, 4, 6, 8]

d_values = range(0, 3)

q_values = range(0, 3)

warnings.filterwarnings("ignore")

evaluate_models(p_values, d_values, q_values)
# ARIMA PDQ Param Tuning said that BEST ARIMA -> (4,1,0)

validation((4,1,0))