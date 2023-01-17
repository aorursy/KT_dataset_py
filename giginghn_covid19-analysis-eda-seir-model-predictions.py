from IPython.display import Image

Image("../input/images/flatten-the-curve.png")
### Load packages



import pandas as pd

import seaborn as sns

import numpy as np

import numpy

import matplotlib.pyplot as plt

from matplotlib import style

from sklearn.model_selection import cross_val_predict

from sklearn import metrics

from sklearn import svm

%matplotlib inline



sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

sns.set_palette("husl")





#Preprocessing

from sklearn import preprocessing

from sklearn.model_selection import train_test_split



#metrics

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, make_scorer



#feature selection

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn import model_selection



#Random Forest

from sklearn.ensemble import RandomForestClassifier

from sklearn import datasets

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor



#Regression

from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression

from sklearn import metrics

import statsmodels.api as sm



#logistic curve

from scipy.optimize import curve_fit



#LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



#seed

import random



# Normalizing continuous variables

from sklearn.preprocessing import MinMaxScaler





# Visualization



## Bokeh

from bokeh.plotting import output_notebook, figure, show

from bokeh.models import ColumnDataSource, Div, Select, Button, ColorBar, CustomJS

from bokeh.layouts import row, column, layout

from bokeh.transform import cumsum, linear_cmap

from bokeh.palettes import Blues8, Spectral3

from bokeh.plotting import figure, output_file, show



## Plotly

from plotly.offline import iplot

from plotly import tools

import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)



from matplotlib import dates

import plotly.graph_objects as go



# Time series

from fbprophet import Prophet

import datetime

from datetime import datetime



# Google BigQuery

from google.cloud import bigquery



#import cdist

from scipy.spatial.distance import cdist



# to solve SEIR

from scipy.integrate import solve_ivp



#others

from pathlib import Path

import os

from tqdm.notebook import tqdm

from scipy.optimize import minimize

from sklearn.metrics import mean_squared_log_error, mean_squared_error

### Load in the data from Kaggle Week 2



train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv",parse_dates=['Date'])

                    

train.tail()
train.info()
# Test dataset from Kaggle



test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv",parse_dates=['Date'])

                    

test.tail()

submit = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')

                     

submit.head()
#read the complete data set



complete_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])



complete_data = complete_data.rename(columns = {'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})



complete_data.sort_values(by=['Date','Confirmed'], ascending=False).head()
complete_data.info()
#read the demographic data



demo_data = pd.read_csv('../input/countryinfo/covid19countryinfo.csv')



demo_data['pop']=demo_data['pop'].str.replace(',', '').astype('float')



demo_data['healthexp']=demo_data['healthexp'].str.replace(',', '').astype('float')





demo_data.head()
pop_info = pd.read_csv('../input/covid19-population-data/population_data.csv')



pop_info.head()
# Weather data

weather_data = pd.read_csv("../input/weather-features/training_data_with_weather_info_week_2.csv", parse_dates=['Date'])

weather_test = pd.read_csv("../input/weather-features/testing_data_with_weather_info_week_2.csv", parse_dates=['Date'])



weather_data.head()

## From @winterpierre source



weather_addition = pd.read_csv("../input/covid19-global-weather-data/temperature_dataframe.csv", parse_dates=['date'])



#rename column

weather_addition.columns = ['Unnamed: 0', 'Id', 'Province_State', 'Country_Region', 'lat', 'long', 'Date',

       'ConfirmedCases', 'Fatalities', 'capital', 'humidity', 'sunHour', 'tempC',

       'windspeedKmph']



#fix the name US for consistency

weather_addition = weather_addition.replace('USA','US')



weather_addition.head()
# case 

case = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Formula: Active Case = Confirmed - Deaths - Recovered

complete_data['Active'] = complete_data['Confirmed'] - complete_data['Deaths'] - complete_data['Recovered']



# impute missing values 

complete_data[['Province_State']] = complete_data[['Province_State']].fillna('')

complete_data[case] = complete_data[case].fillna(0)



complete_data.sort_values(by=['Date','Confirmed'], ascending=False).head()
complete_data.sort_values(by=['Date'], ascending=False).tail()
map_covid = train.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()

map_covid['Date'] = map_covid['Date'].dt.strftime('%m/%d/%Y')

map_covid['size'] = map_covid['ConfirmedCases'].pow(0.3) * 3.5



fig = px.scatter_geo(map_covid, locations="Country_Region", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country_Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Confirmed Cases Around the Globe', color_continuous_scale="tealrose")

fig.show()

regions =pd.DataFrame()



regions['Country'] = map_covid["Country_Region"]

regions['Confirmed Cases'] = map_covid["ConfirmedCases"]



fig = px.choropleth(regions, locations='Country',

                    locationmode='country names',

                    color="Confirmed Cases",color_continuous_scale="tealrose")



fig.update_layout(title="COVID19 Confirmed Cases on 04-01-2020")



fig.show()

# sum of all Confirmed cases by country as of March 26

sum_confirm = pd.DataFrame(complete_data.loc[complete_data['Date']==complete_data['Date'].max()].groupby(

    ['Country_Region'])['Confirmed'].sum()).reset_index()



# sum of all Death cases by country as of March 26

sum_death = pd.DataFrame(complete_data.loc[complete_data['Date']==complete_data['Date'].max()].groupby(

    ['Country_Region'])['Deaths'].sum()).reset_index()



# sum of all Recovered cases by country as of March 26

sum_recover = pd.DataFrame(complete_data.loc[complete_data['Date']==complete_data['Date'].max()].groupby(

    ['Country_Region'])['Recovered'].sum()).reset_index()



# sum of all Active cases by country as of March 26

sum_active = pd.DataFrame(complete_data.loc[complete_data['Date']==complete_data['Date'].max()].groupby(

    ['Country_Region'])['Active'].sum()).reset_index()

sns.set(rc={'figure.figsize':(15, 7)})



top20_confirm = sum_confirm.sort_values(by=['Confirmed'], ascending=False).head(20)



plot1 = sns.barplot(x="Confirmed",y="Country_Region", data=top20_confirm)



plt.title("Total Numbers of Confirmed Cases",fontsize=20)



for p in plot1.patches:

    width = p.get_width()

    plot1.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")
top20_confirm['Country_Region'].unique()


top20_death = sum_death.sort_values(by=['Deaths'], ascending=False).head(20)



plot2 = sns.barplot(x="Deaths",y="Country_Region", data=top20_death)



plt.title("Total Numbers of Fatal Cases",fontsize=20)



for p in plot2.patches:

    width = p.get_width()

    plot2.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")
top20_recover = sum_recover.sort_values(by=['Recovered'], ascending=False).head(20)



plot3 = sns.barplot(x="Recovered",y="Country_Region", data=top20_recover)



plt.title("Total Numbers of Recovered Cases",fontsize=20)



for p in plot3.patches:

    width = p.get_width()

    plot3.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")
top20_active = sum_active.sort_values(by=['Active'], ascending=False).head(20)



plot4 = sns.barplot(x="Active",y="Country_Region", data=top20_active)



plt.title("Total Numbers of Active Cases",fontsize=20)



for p in plot4.patches:

    width = p.get_width()

    plot4.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")
### Compute day first outbreak for each country

complete_data_first = train.copy()



countries_array = complete_data_first['Country_Region'].unique()



complete_data_first_outbreak = pd.DataFrame()



for i in countries_array:

    # get relevant data 

    day_first_outbreak = complete_data_first.loc[complete_data_first['Country_Region']==i]

    

    date_outbreak = day_first_outbreak.loc[day_first_outbreak['ConfirmedCases']>0]['Date'].min()

    

    #Calculate days since first outbreak happened

    day_first_outbreak['days_since_first_outbreak'] = (day_first_outbreak['Date'] 

                                                       - date_outbreak).astype('timedelta64[D]')

    

    #impute the negative days with 0

    day_first_outbreak['days_since_first_outbreak'][day_first_outbreak['days_since_first_outbreak']<0] = 0 

   

    complete_data_first_outbreak = complete_data_first_outbreak.append(day_first_outbreak,ignore_index=True)

    



complete_data_first_outbreak.head()

top20_confirm_first = complete_data_first_outbreak.loc[

    complete_data_first_outbreak['Country_Region'].isin(

        ['US', 'China', 'Italy', 'Spain', 'Germany', 'France', 'Iran',

       'United Kingdom', 'Switzerland', 'Korea, South'])==True]



top20_confirm_first = top20_confirm_first.groupby(['Country_Region','days_since_first_outbreak'])['ConfirmedCases'].sum().reset_index()



top20_confirm_first['days_since_first_outbreak'] = pd.to_timedelta(

    top20_confirm_first['days_since_first_outbreak'], unit='D')



sns.lineplot(data=top20_confirm_first, x="days_since_first_outbreak", y="ConfirmedCases", hue="Country_Region")



plt.ylabel("Total cases in top 10 countries")



plt.xlabel("Number of days since first outbreak")





plt.title("Total numbers of cases since first outbreak in top 10 countries",fontsize=20)





time_sum = complete_data.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()



time_sum = pd.melt(time_sum, id_vars=['Date'], value_vars=['Confirmed','Deaths'])



time_sum = time_sum.rename(columns={"value": "total", "variable": "Cases"})



sns.lineplot(data=time_sum, x="Date", y="total", hue="Cases")



plt.ylabel("Total cases")



plt.title("Total numbers of Cases",fontsize=20)

sns.set(rc={'figure.figsize':(15, 7)})



time_log = complete_data.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()



time_log["Confirmed"] = np.log(time_log["Confirmed"])



time_log["Deaths"] = np.log(time_log["Deaths"])



time_log = pd.melt(time_log, id_vars=['Date'], value_vars=['Confirmed','Deaths'])



time_log = time_log.rename(columns={"value": "total", "variable": "Cases"})



sns.lineplot(data=time_log, x="Date", y="total", hue="Cases")



plt.ylabel("Total Confirmed cases on log scale")



plt.title("Total numbers of Confirmed on log scale",fontsize=20)

china_sum = complete_data.loc[complete_data['Country_Region']=="China"].groupby('Date')['Date', 'Confirmed','Deaths','Active','Recovered'].sum().reset_index()



china_sum = pd.melt(china_sum, id_vars=['Date'], value_vars=['Confirmed','Deaths','Active','Recovered'])



china_sum = china_sum.rename(columns={"value": "total", "variable": "Cases"})



sns.lineplot(data=china_sum, x="Date", y="total", hue="Cases")



plt.ylabel("Total cases in China")



plt.title("Total numbers of Cases in China",fontsize=20)

italy_sum = complete_data.loc[complete_data['Country_Region']=="Italy"].groupby('Date')['Date', 'Confirmed','Deaths','Active','Recovered'].sum().reset_index()



italy_sum = pd.melt(italy_sum, id_vars=['Date'], value_vars=['Confirmed','Deaths','Active','Recovered'])



italy_sum = italy_sum.rename(columns={"value": "total", "variable": "Cases"})



sns.lineplot(data=italy_sum, x="Date", y="total", hue="Cases")



plt.ylabel("Total cases in Italy")



plt.title("Total numbers of Cases in Italy",fontsize=20)

us_sum = complete_data.loc[complete_data['Country_Region']=="US"].groupby('Date')['Date', 'Confirmed','Deaths','Active','Recovered'].sum().reset_index()



us_sum = pd.melt(us_sum, id_vars=['Date'], value_vars=['Confirmed','Deaths','Active','Recovered'])



us_sum = us_sum.rename(columns={"value": "total", "variable": "Cases"})



sns.lineplot(data=us_sum, x="Date", y="total", hue="Cases")



plt.ylabel("Total cases in the U.S.")



plt.title("Total numbers of Cases in the U.S.",fontsize=20)

#Other countries



other_sum = complete_data.loc[complete_data['Country_Region'].isin(["Italy","China","US"])==False]



other_sum = other_sum.groupby('Date')['Date', 'Confirmed','Deaths','Active','Recovered'].sum().reset_index()



other_sum = pd.melt(other_sum, id_vars=['Date'], value_vars=['Confirmed','Deaths','Active','Recovered'])



other_sum = other_sum.rename(columns={"value": "total", "variable": "Cases"})



sns.lineplot(data=other_sum, x="Date", y="total", hue="Cases")



plt.ylabel("Total cases in other countries")



plt.title("Total numbers of Cases in other countries",fontsize=20)

# Countries that are in Europe



europe = ['Austria','Italy','Belgium','Latvia','Bulgaria','Lithuania','Croatia','Luxembourg',

          'Cyprus','Malta','Czechia','Netherlands','Denmark','Poland','Estonia','Portugal',

          'Finland','Romania','France','Slovakia','Germany','Slovenia','Greece','Spain',

          'Hungary','Sweden','Ireland','Switzerland','United Kingdom']



europe_sum = complete_data.loc[complete_data['Country_Region'].isin(europe)==True]



europe_sum.loc[europe_sum['Confirmed']>0].sort_values('Date').head(1)

#Plot out the total cases by each country

europe_sum = europe_sum.loc[complete_data['Date']==complete_data['Date'].max()].groupby(

    'Country_Region')['Country_Region', 'Confirmed'].sum().reset_index().sort_values('Confirmed',ascending=False)



plot5 = sns.barplot(x="Confirmed",y="Country_Region", data=europe_sum)



plt.title("Total Numbers of Confirmed Cases")



for p in plot1.patches:

    width = p.get_width()

    plot1.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")

top10_eu_sum = complete_data.loc[complete_data['Country_Region'].isin(['Italy','Spain','Germany',

                                                                      'France','United Kingdom',

                                                                      'Switzerland','Netherlands','Austria',

                                                                      'Belgium','Portugal'])==True]



top10_eu_sum1 = top10_eu_sum.groupby(['Country_Region','Date'])['Confirmed'].sum().reset_index()



sns.lineplot(data=top10_eu_sum1, x="Date", y="Confirmed", hue="Country_Region")



plt.ylabel("Total cases in Europe countries")



plt.title("Total numbers of Cases in Europe countries",fontsize=20)

top10_eu_sum2 = top10_eu_sum.groupby(['Country_Region','Date'])['Confirmed','Deaths'].sum().reset_index()



top10_eu_sum2['Fatal_Rate'] = round((top10_eu_sum2['Deaths']/top10_eu_sum2['Confirmed'])*100,2)



sns.lineplot(data=top10_eu_sum2, x="Date", y="Fatal_Rate", hue="Country_Region")



plt.ylabel("Fatality Rate in Europe countries in Percentage")



plt.title("Fatality Rate  in Europe countries",fontsize=20)
top10_eu_sum3 = top10_eu_sum.groupby(['Country_Region','Date'])['Confirmed','Recovered'].sum().reset_index()



top10_eu_sum3['Recover_Rate'] = round((top10_eu_sum3['Recovered']/top10_eu_sum3['Confirmed'])*100,2)



sns.lineplot(data=top10_eu_sum3, x="Date", y="Recover_Rate", hue="Country_Region")



plt.ylabel("Recovery Rate in Europe countries in Percentage")



plt.title("Recovery Rate  in Europe countries",fontsize=20)
north_america = ['Antigua and Barbuda','Bahamas','Barbados','Belize','Canada','Costa Rica','Cuba','El Salvador',

                 'Grenada','Guatemala','Haití','Honduras','Jamaica','Mexico','Nicaragua','Panama',

                 'Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines','Trinidad and Tobago','US']



na_region_sum = complete_data.loc[complete_data['Country_Region'].isin(north_america)==True]



na_region_sum.loc[na_region_sum['Confirmed']>0].sort_values('Date').head(1)

# plot the total of US

us_region_sum = train.loc[train['Country_Region'] == "US"]



us_region_sum1 = us_region_sum.loc[us_region_sum['Date']==train['Date'].max()].groupby(

    ['Province_State'])['ConfirmedCases'].sum().reset_index().sort_values('ConfirmedCases',ascending=False).head(20)



plot6 = sns.barplot(x="ConfirmedCases",y="Province_State", data=us_region_sum1)



plt.ylabel("Total confirmed cases by US states")



plt.title("Total Numbers of Confirmed Cases by U.S top 20 states",fontsize=20)



for p in plot6.patches:

    width = p.get_width()

    plot6.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")

    





top10_us_sum = us_region_sum.loc[us_region_sum['Province_State'].isin(['New York','New Jersey','Washington',

                                                                      'California','Michigan',

                                                                      'Illinois','Florida','Louisiana',

                                                                      'Pennsylvania','Texas'])==True]





top10_us_sum1 = top10_us_sum.groupby(

    ['Province_State','Date'])['ConfirmedCases'].sum().reset_index()

    

sns.lineplot(data=top10_us_sum1, x="Date", y="ConfirmedCases", hue="Province_State")



plt.ylabel("Total confirmed cases by US states")



plt.title("Total numbers of Confirmed Cases in by top 10 states",fontsize=20)

top10_us_sum2 = top10_us_sum.groupby(['Province_State','Date'])['ConfirmedCases','Fatalities'].sum().reset_index()



top10_us_sum2['Fatal_Rate'] = round((top10_us_sum2['Fatalities']/top10_us_sum2['ConfirmedCases'])*100,2)



sns.lineplot(data=top10_us_sum2, x="Date", y="Fatal_Rate", hue="Province_State")



plt.ylabel("Fatality Rate in the U.S. states in Percentage")



plt.title("Fatality Rate by top 10 states",fontsize=20)
top10_us_sum3 = complete_data.loc[complete_data['Country_Region']=='US'].groupby(

    'Date')['Confirmed','Recovered'].sum().reset_index()



top10_us_sum3['Recover_Rate'] = round((top10_us_sum3['Recovered']/top10_us_sum3['Confirmed'])*100,2)



sns.lineplot(data=top10_us_sum3, x="Date", y="Recover_Rate")



plt.ylabel("Recovery Rate in the U.S. states in Percentage")



plt.title("Recovery Rate by top 10 states",fontsize=20)
round(top10_us_sum3['Recover_Rate'].mean(),2)


asia = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 

        'Cambodia', 'China', 'Timor-Leste', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 

        'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 

        'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'Oman', 'Pakistan', 

        'Philippines', 'Qatar', 'Russia', 'Saudi Arabia', 'Singapore', 'Korea, South', 'Sri Lanka', 

        'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 

        'Uzbekistan', 'Vietnam', 'Yemen']



asia_sum = train.loc[train['Country_Region'].isin(asia)==True]



asia_sum1 = asia_sum.loc[asia_sum['Date']==train['Date'].max()].groupby(

    'Country_Region')['Country_Region', 'ConfirmedCases'].sum().reset_index().sort_values(

    'ConfirmedCases',ascending=False).head(20)



plot7 = sns.barplot(x="ConfirmedCases",y="Country_Region", data=asia_sum1)



plt.title("Total Numbers of Confirmed Cases",fontsize=20)



for p in plot7.patches:

    width = p.get_width()

    plot7.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")





#Let's plot them in a timeline but excluding China



top10_asia_sum1 = asia_sum.loc[asia_sum['Country_Region'].isin(['Korea, South','Iran','Turkey','Israel',

                                                               'Malaysia','Japan','Pakistan',

                                                               'Thailand','Saudi Arabia','Indonesia',

                                                              'Russia','India'])==True]





top10_asia_sum1 = top10_asia_sum1.groupby(

    ['Country_Region','Date'])['ConfirmedCases'].sum().reset_index()

    

sns.lineplot(data=top10_asia_sum1, x="Date", y="ConfirmedCases", hue="Country_Region")



plt.ylabel("Total confirmed cases by Asia countries")



plt.title("Total numbers of Confirmed Cases in by Asia countries - excluding China",fontsize=20)

#excluding Iran, Korea and China



top10_asia_sum2 = asia_sum.loc[asia_sum['Country_Region'].isin(['Turkey','Israel',

                                                               'Malaysia','Japan','Pakistan',

                                                               'Thailand','Saudi Arabia','Indonesia',

                                                              'Russia','India','Philippines'])==True]





top10_asia_sum2 = top10_asia_sum2.groupby(

    ['Country_Region','Date'])['ConfirmedCases'].sum().reset_index()

    

sns.lineplot(data=top10_asia_sum2, x="Date", y="ConfirmedCases", hue="Country_Region")



plt.ylabel("Total confirmed cases by Asia countries")



plt.title("Total numbers of Confirmed Cases in by Asia countries - excluding China, Korea and Iran",fontsize=20)


top10_asia_sum3 = asia_sum.loc[asia_sum['Country_Region'].isin(['Korea, South','Turkey','Israel',

                                                               'Malaysia','Japan','Pakistan',

                                                               'Thailand','Saudi Arabia','Indonesia',

                                                              'Russia','India'])==True]





top10_asia_sum3 = top10_asia_sum3.groupby(

    ['Country_Region','Date'])['ConfirmedCases','Fatalities'].sum().reset_index()



top10_asia_sum3['Fatal_Rate'] = round((top10_asia_sum3['Fatalities']/top10_asia_sum3['ConfirmedCases'])*100,2)

    

sns.lineplot(data=top10_asia_sum3, x="Date", y="Fatal_Rate", hue="Country_Region")



plt.ylabel("Fatality Rate in Asia countries in Percentage")



plt.title("Fatality Rate  in Asia countries - excluding China and Iran",fontsize=20)


top10_asia_sum4 = complete_data.loc[complete_data['Country_Region'].isin(['Korea, South','Iran','Turkey','Israel',

                                                               'Malaysia','Japan','Pakistan',

                                                               'Thailand','Saudi Arabia','Indonesia',

                                                              'Russia','India','Philippines'])==True]



top10_asia_sum4 = top10_asia_sum4.groupby(['Country_Region','Date'])['Confirmed','Recovered'].sum().reset_index()



top10_asia_sum4['Recover_Rate'] = round((top10_asia_sum4['Recovered']/top10_asia_sum4['Confirmed'])*100,2)



sns.lineplot(data=top10_asia_sum4, x="Date", y="Recover_Rate", hue="Country_Region")



plt.ylabel("Recovery Rate in Europe countries in Percentage")



plt.title("Recovery Rate  in Europe countries",fontsize=20)


temp_covid = weather_data.groupby(['Date', 'Country_Region'])['temp'].mean().reset_index()

temp_covid['Date'] = pd.to_datetime(temp_covid['Date'])

map_covid['Date'] = pd.to_datetime(map_covid['Date'])



#merge with the confirmed cases for size changing

temp_covid = pd.merge(temp_covid, map_covid, on=['Date','Country_Region'],how='left')



temp_covid['Date'] = temp_covid['Date'].dt.strftime('%m/%d/%Y')



fig = px.scatter_geo(temp_covid, locations="Country_Region", locationmode='country names', 

                     color="temp", size='size', hover_name="Country_Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Temperature according to the number of Confirmed Cases Around the Globe', 

                     color_continuous_scale="tealrose")

fig.show()

wdsp_covid = weather_data.groupby(['Date', 'Country_Region'])['wdsp'].max().reset_index()

wdsp_covid['Date'] = pd.to_datetime(wdsp_covid['Date'])



#merge with the confirmed cases for size changing

wdsp_covid = pd.merge(wdsp_covid, map_covid, on=['Date','Country_Region'],how='left')



wdsp_covid['Date'] = wdsp_covid['Date'].dt.strftime('%m/%d/%Y')



fig = px.scatter_geo(wdsp_covid, locations="Country_Region", locationmode='country names', 

                     color="wdsp", size='size', hover_name="Country_Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Windspeed according to the number of Confirmed Cases Around the Globe', 

                     color_continuous_scale="tealrose")

fig.show()

humid_covid = weather_addition.groupby(['Date', 'Country_Region'])['humidity'].mean().reset_index()

humid_covid['Date'] = pd.to_datetime(humid_covid['Date'])



#merge with the confirmed cases for size changing

humid_covid = pd.merge(humid_covid, map_covid, on=['Date','Country_Region'],how='left')

humid_covid = humid_covid.dropna()



humid_covid['Date'] = humid_covid['Date'].dt.strftime('%m/%d/%Y')



fig = px.scatter_geo(humid_covid, locations="Country_Region", locationmode='country names', 

                     color="humidity", size='size', hover_name="Country_Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Humidity according to the number of Confirmed Cases Around the Globe', 

                     color_continuous_scale="tealrose")

fig.show()

sun_covid = weather_addition.groupby(['Date', 'Country_Region'])['sunHour'].mean().reset_index()

sun_covid['Date'] = pd.to_datetime(sun_covid['Date'])



#merge with the confirmed cases for size changing

sun_covid = pd.merge(sun_covid, map_covid, on=['Date','Country_Region'],how='left')

sun_covid = sun_covid.dropna()



sun_covid['Date'] = sun_covid['Date'].dt.strftime('%m/%d/%Y')



fig = px.scatter_geo(sun_covid, locations="Country_Region", locationmode='country names', 

                     color="sunHour", size='size', hover_name="Country_Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Humidity according to the number of Confirmed Cases Around the Globe', 

                     color_continuous_scale="tealrose")

fig.show()

#join the dataframe



temp_covid1 = pd.merge(temp_covid, wdsp_covid[['Date','Country_Region','wdsp']], 

                             on=['Date','Country_Region'],how='left')

temp_covid1 = pd.merge(temp_covid1, humid_covid[['Date','Country_Region','humidity']], 

                             on=['Date','Country_Region'],how='left')

temp_covid1 = pd.merge(temp_covid1, sun_covid[['Date','Country_Region','sunHour']], 

                             on=['Date','Country_Region'],how='left')



temp_covid1 = temp_covid1.dropna()



temp_covid1.head()

#construct the Multilinear regression model

X =  temp_covid1[['temp','wdsp', 'humidity', 'sunHour']]

y = temp_covid1['ConfirmedCases']



# Note the difference in argument order

model = sm.OLS(y, X).fit()



#model summary

model.summary()

china_temp = temp_covid1.loc[temp_covid1['Country_Region']=='China']



#construct the OLS model

X =  china_temp[['temp', 'wdsp', 'humidity', 'sunHour']]

y = china_temp['ConfirmedCases']



# Note the difference in argument order

model = sm.OLS(y, X).fit()



#predictions = model.predict(X) # make the predictions by the model



model.summary()

italy_temp = temp_covid1.loc[temp_covid1['Country_Region']=='Italy']



#construct the OLS model

X =  italy_temp[['temp', 'wdsp', 'humidity', 'sunHour']]

y = italy_temp['ConfirmedCases']



# Note the difference in argument order

model = sm.OLS(y, X).fit()



#predictions = model.predict(X) # make the predictions by the model



model.summary()


us_temp = temp_covid1.loc[temp_covid1['Country_Region']=='US']



#construct the OLS model

X =  us_temp[['temp','wdsp', 'humidity', 'sunHour']]

y = us_temp['ConfirmedCases']



# Note the difference in argument order

model = sm.OLS(y, X).fit()



#predictions = model.predict(X) # make the predictions by the model



model.summary()

#Combine US states to only US 

demo_data1 = demo_data.replace(['Alabama', 'Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',

             'Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',

             'Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska',

             'Nevada','New Hampshire','New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',

             'Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota', 'Tennessee',

             'Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming','San Franciso',

             'GeorgiaUS', 'Atlanta', 'Honolulu', 'Washington DC'], 'US')



demo_data1.head()



demo_data_pop = demo_data1.groupby(['country'])['country','pop'].sum().reset_index().sort_values('pop',ascending=False)



demo_data_pop = demo_data_pop.loc[demo_data_pop['country'].isin(['US', 'Italy', 'China', 'Spain', 'Germany', 'France', 'Iran',

       'United Kingdom', 'Switzerland', 'Netherlands', 'Belgium',

       'Korea, South', 'Turkey', 'Austria', 'Canada', 'Portugal', 'Norway',

       'Brazil', 'Israel', 'Australia'])==True]



plot_pop=sns.barplot(x="pop",y="country", data=demo_data_pop)



plt.xlabel("Population")



plt.title("Population",fontsize=20)



for p in plot_pop.patches:

    width = p.get_width()

    plot_pop.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")

    


demo_data2 = demo_data1.sort_values('tests',ascending=False).head(20)



plot10 = sns.barplot(x="tests",y="country", data=demo_data2)



plt.xlabel("Total number of COVID-19 test")



plt.title("Total number of COVID-19 test",fontsize=20)



for p in plot10.patches:

    width = p.get_width()

    plot10.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")


demo_data3 = demo_data1.groupby(['country'])['country',

                                           'hospibed'].mean().reset_index().sort_values('hospibed',ascending=False)



demo_data3 = demo_data3.loc[demo_data3['country'].isin(['US', 'Italy', 'China', 'Spain', 'Germany', 'France', 'Iran',

       'United Kingdom', 'Switzerland', 'Netherlands', 'Belgium',

       'Korea, South', 'Turkey', 'Austria', 'Canada', 'Portugal', 'Norway',

       'Brazil', 'Israel', 'Australia'])==True]



plot11 = sns.barplot(x="hospibed",y="country", data=demo_data3)



plt.xlabel("Hospital bed per 1,000 people")



plt.title("Amount of hospital bed per 1,000 people",fontsize=20)

demo_data9 = demo_data1.groupby(['country'])['country',

                                           'medianage'].median().reset_index().sort_values('medianage',ascending=False)



demo_data9 = demo_data9.loc[demo_data9['country'].isin(['US', 'Italy', 'China', 'Spain', 'Germany', 'France', 'Iran',

       'United Kingdom', 'Switzerland', 'Netherlands', 'Belgium',

       'Korea, South', 'Turkey', 'Austria', 'Canada', 'Portugal', 'Norway',

       'Brazil', 'Israel', 'Australia'])==True]



plot13 = sns.barplot(x="medianage",y="country", data=demo_data9)



plt.xlabel("Median Age")



plt.title("Median Age by Country",fontsize=20)
demo_data4 = demo_data1.groupby(['country'])['country',

                                           'density'].sum().reset_index().sort_values('density',ascending=False)



demo_data4 = demo_data4.loc[demo_data4['country'].isin(['US', 'Italy', 'China', 'Spain', 'Germany', 'France', 'Iran',

       'United Kingdom', 'Switzerland', 'Netherlands', 'Belgium',

       'Korea, South', 'Turkey', 'Austria', 'Canada', 'Portugal', 'Norway',

       'Brazil', 'Israel', 'Australia'])==True]



plot12 = sns.barplot(x="density",y="country", data=demo_data4)



plt.xlabel("denisity")



plt.title("Population Density by Country",fontsize=20)



for p in plot12.patches:

    width = p.get_width()

    plot12.text(width + 1.5  ,

            p.get_y()+p.get_height()/2. + 0.2,

            '{:1.0f}'.format(width),

            ha="left")

    
demo_data7 = demo_data1.groupby(['country'])['country',

                                           'lung'].mean().reset_index().sort_values('lung',ascending=False)



demo_data7 = demo_data7.loc[demo_data7['country'].isin(['US', 'Italy', 'China', 'Spain', 'Germany', 'France', 'Iran',

       'United Kingdom', 'Switzerland', 'Netherlands', 'Belgium',

       'Korea, South', 'Turkey', 'Austria', 'Canada', 'Portugal', 'Norway',

       'Brazil', 'Israel', 'Australia'])==True]



plot14 = sns.barplot(x="lung",y="country", data=demo_data7)



plt.xlabel("Death rate from lung diseases per 100k people")



plt.title("Death rate from lung diseases per 100k people by Country",fontsize=20)

demo_data11 = demo_data1.groupby(['country'])['country',

                                           'smokers'].sum().reset_index().sort_values('smokers',ascending=False)



demo_data11 = demo_data11.loc[demo_data11['country'].isin(['US', 'Italy', 'China', 'Spain', 'Germany', 'France', 'Iran',

       'United Kingdom', 'Switzerland', 'Netherlands', 'Belgium',

       'Korea, South', 'Turkey', 'Austria', 'Canada', 'Portugal', 'Norway',

       'Brazil', 'Israel', 'Australia'])==True]



plot111 = sns.barplot(x="smokers",y="country", data=demo_data11)



plt.xlabel("Number of smokers")



plt.title("Number of smokers by Country",fontsize=20)


china_sum = complete_data.loc[complete_data['Country_Region']=="China"].groupby('Date')['Date', 'Confirmed','Deaths','Active','Recovered'].sum().reset_index()



china_sum = pd.melt(china_sum, id_vars=['Date'], value_vars=['Confirmed','Deaths','Active','Recovered'])



china_sum = china_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=china_sum, x="Date", y="total", hue="Cases")



ax.axvline(pd.to_datetime('2020-01-23'), color="red", linestyle="--")



ax.axvline(pd.to_datetime('2020-02-12'), color="gray", linestyle="--")



ax.annotate("Date first quarantine", xy=(pd.to_datetime('2020-01-24'), 50000))



plt.ylabel("Total cases in China")



plt.title("Total numbers of Cases in China",fontsize=20)

Image("../input/images/Epidemic Curve of the Confirmed Cases of Coronavirus Disease 2019 (COVID-19).png")
italy_sum = complete_data.loc[complete_data['Country_Region']=="Italy"].groupby('Date')['Date', 'Confirmed','Deaths','Active','Recovered'].sum().reset_index()



italy_sum = pd.melt(italy_sum, id_vars=['Date'], value_vars=['Confirmed','Deaths','Active','Recovered'])



italy_sum = italy_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=italy_sum, x="Date", y="total", hue="Cases")



ax.axvline(pd.to_datetime('2020-03-19'), color="red", linestyle="--")



ax.annotate("Date first quarantine", xy=(pd.to_datetime('2020-03-20'), 20000))





plt.ylabel("Total cases in Italy")



plt.title("Total numbers of Cases in Italy",fontsize=20)

us_sum = complete_data.loc[complete_data['Country_Region']=="US"].groupby('Date')['Date', 'Confirmed','Deaths','Active','Recovered'].sum().reset_index()



us_sum = pd.melt(us_sum, id_vars=['Date'], value_vars=['Confirmed','Deaths','Active','Recovered'])



us_sum = us_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=us_sum, x="Date", y="total", hue="Cases")



ax.axvline(pd.to_datetime('2020-03-16'), color="red", linestyle="--")



ax.annotate("Date first quarantine", xy=(pd.to_datetime('2020-03-17'), 120000))



plt.ylabel("Total cases in the U.S.")



plt.title("Total numbers of Cases in the U.S.",fontsize=20)

demo_data = demo_data.rename(columns={'country':'Country_Region'})



demo_data_join = demo_data.copy()



demo_data_join = demo_data_join.drop_duplicates(['Country_Region'],keep='first')



train_demo = pd.merge(train, demo_data_join[['Country_Region', 'pop', 'tests',

       'testpop', 'density', 'medianage', 'urbanpop', 

       'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64',

       'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility']], on=['Country_Region'], how='left')



test_demo = pd.merge(test, demo_data_join[['Country_Region', 'pop', 'tests',

       'testpop', 'density', 'medianage', 'urbanpop', 

       'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64',

       'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility']], on=['Country_Region'], how='left')



train_demo.head()
Image("../input/images/log-curve.png")
china_sum = complete_data.loc[complete_data['Country_Region']=="China"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



china_sum = pd.melt(china_sum, id_vars=['Date'], value_vars=['Confirmed'])



china_sum = china_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=china_sum, x="Date", y="total", hue="Cases")



ax.axvline(pd.to_datetime('2020-01-23'), color="red", linestyle="--")



ax.axvline(pd.to_datetime('2020-02-12'), color="gray", linestyle="--")



ax.annotate("Date first quarantine", xy=(pd.to_datetime('2020-01-24'), 50000))



plt.ylabel("Total cases in China")



plt.title("Total numbers of Cases in China",fontsize=20)

china_sum = complete_data.loc[complete_data['Country_Region']=="South Korea"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



china_sum = pd.melt(china_sum, id_vars=['Date'], value_vars=['Confirmed'])



china_sum = china_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=china_sum, x="Date", y="total", hue="Cases")



ax.axvline(pd.to_datetime('2020-03-01'), color="gray", linestyle="--")



ax.axvline(pd.to_datetime('2020-02-01'), color="red", linestyle="--")



ax.annotate("First roll out of widespread testing \n around this time", xy=(pd.to_datetime('2020-02-02'), 5000))



plt.ylabel("Total confirmed cases in South Korea")



plt.title("Total numbers of Cases in South Korea",fontsize=20)

# This functions smooths data, thanks to Dan Pearson. We will use it to smooth the data for growth factor.

def smoother(inputdata,w,imax):

    data = 1.0*inputdata

    data = data.replace(np.nan,1)

    data = data.replace(np.inf,1)

    #print(data)

    smoothed = 1.0*data

    normalization = 1

    for i in range(-imax,imax+1):

        if i==0:

            continue

        smoothed += (w**abs(i))*data.shift(i,axis=0)

        normalization += w**abs(i)

    smoothed /= normalization

    return smoothed



def growth_factor(confirmed):

    confirmed_iminus1 = confirmed.shift(1, axis=0)

    confirmed_iminus2 = confirmed.shift(2, axis=0)

    return (confirmed-confirmed_iminus1)/(confirmed_iminus1-confirmed_iminus2)



def growth_ratio(confirmed):

    confirmed_iminus1 = confirmed.shift(1, axis=0)

    return (confirmed/confirmed_iminus1)



# This is a function which plots (for in input country) the active, confirmed, and recovered cases, deaths, and the growth factor.

def plot_country_active_confirmed_recovered(country):

    

    # Plots Active, Confirmed, and Recovered Cases. Also plots deaths.

    country_data = train[train['Country_Region']==country]

    table = country_data.drop(['Id','Province_State'], axis=1)



    table2 = pd.pivot_table(table, values=['ConfirmedCases','Fatalities'], index=['Date'], aggfunc=np.sum)

    table3 = table2.drop(['Fatalities'], axis=1)

   

    # Growth Factor

    w = 0.5

    table2['GrowthFactor'] = growth_factor(table2['ConfirmedCases'])

    table2['GrowthFactor'] = smoother(table2['GrowthFactor'],w,5)



    # 2nd Derivative

    table2['2nd_Derivative'] = np.gradient(np.gradient(table2['ConfirmedCases'])) #2nd derivative

    table2['2nd_Derivative'] = smoother(table2['2nd_Derivative'],w,7)





    #Plot confirmed[i]/confirmed[i-1], this is called the growth ratio

    table2['GrowthRatio'] = growth_ratio(table2['ConfirmedCases'])

    table2['GrowthRatio'] = smoother(table2['GrowthRatio'],w,5)

    

        #Plot the growth rate, we will define this as k in the logistic function presented at the beginning of this notebook.

    table2['GrowthRate']=np.gradient(np.log(table2['ConfirmedCases']))

    table2['GrowthRate'] = smoother(table2['GrowthRate'],0.5,3)

    

    # horizontal line at growth rate 1.0 for reference

    x_coordinates = [1, 100]

    y_coordinates = [1, 1]

    

    sns.set(rc={'figure.figsize':(10, 5)})



    

    pd.plotting.register_matplotlib_converters()



    #plots

    table2['Fatalities'].plot(title='Fatalities')

    plt.show()

    table3.plot() 

    plt.show()

    table2['GrowthFactor'].plot(title='Growth Factor')

    plt.plot(x_coordinates, y_coordinates) 

    plt.show()

    table2['2nd_Derivative'].plot(title='2nd_Derivative')

    plt.show()

    table2['GrowthRatio'].plot(title='Growth Ratio')

    plt.plot(x_coordinates, y_coordinates)

    plt.show()

    table2['GrowthRate'].plot(title='Growth Rate')

    plt.show()





    return 

plot_country_active_confirmed_recovered('China')
plot_country_active_confirmed_recovered('Korea, South')
plot_country_active_confirmed_recovered('US')
plot_country_active_confirmed_recovered('Italy')
plot_country_active_confirmed_recovered('Spain')
plot_country_active_confirmed_recovered('Vietnam')
Image("../input/images/SEIR-math.png")

# Function code refernece from https://www.kaggle.com/anjum48/seir-model-with-intervention



# Susceptible equation

def dS_dt(S, I, R_t, T_inf):

    return -(R_t / T_inf) * I * S



# Exposed equation

def dE_dt(S, E, I, R_t, T_inf, T_inc):

    return (R_t / T_inf) * I * S - (T_inc**-1) * E



# Infected equation

def dI_dt(I, E, T_inc, T_inf):

    return (T_inc**-1) * E - (T_inf**-1) * I



# Recovered/Remove/deceased equation

def dR_dt(I, T_inf):

    return (T_inf**-1) * I



def SEIR_model(t, y, R_t, T_inf, T_inc):

    

    if callable(R_t):

        reproduction = R_t(t)

    else:

        reproduction = R_t

        

    S, E, I, R = y

    

    S_out = dS_dt(S, I, reproduction, T_inf)

    E_out = dE_dt(S, E, I, reproduction, T_inf, T_inc)

    I_out = dI_dt(I, E, T_inc, T_inf)

    R_out = dR_dt(I, T_inf)

    

    return [S_out, E_out, I_out, R_out]
## Thanks @funkyboy for the plotting function



def plot_model_and_predict(data, pop, solution, title='SEIR model'):

    sus, exp, inf, rec = solution.y

    

    f = plt.figure(figsize=(16,5))

    ax = f.add_subplot(1,2,1)

    #ax.plot(sus, 'b', label='Susceptible');

    ax.plot(exp, 'y', label='Exposed');

    ax.plot(inf, 'r', label='Infected');

    ax.plot(rec, 'c', label='Recovered/deceased');

    plt.title(title)

    plt.xlabel("Days", fontsize=10);

    plt.ylabel("Fraction of population", fontsize=10);

    plt.legend(loc='best');

    

    ax2 = f.add_subplot(1,2,2)

    preds = np.clip((inf + rec) * pop ,0,np.inf)

    ax2.plot(range(len(data)),preds[:len(data)],label = 'Predict ConfirmedCases')

    ax2.plot(range(len(data)),data['ConfirmedCases'])

    plt.title('Model predict and data')

    plt.ylabel("Population", fontsize=10);

    plt.xlabel("Days", fontsize=10);

    plt.legend(loc='best');
Country = 'New York'

N = pop_info[pop_info['Name']==Country]['Population'].tolist()[0] # Hubei Population 



# Load dataset of Hubei

train_loc = train[train['Country_Region']==Country].query('ConfirmedCases > 0')

if len(train_loc)==0:

    train_loc = train[train['Province_State']==Country].query('ConfirmedCases > 0')



n_infected = train_loc['ConfirmedCases'].iloc[0] # start from first comfirmedcase on dataset first date

max_days = len(train_loc)# how many days want to predict



# Initial stat for SEIR model

s = (N - n_infected)/ N

e = 0.

i = n_infected / N

r = 0.



# Define all variable of SEIR model 

T_inc = 5.2  # average incubation period

T_inf = 2.9 # average infectious period

R_0 = 3.954 # reproduction number



## Solve the SEIR model 

sol = solve_ivp(SEIR_model, [0, max_days], [s, e, i, r], args=(R_0, T_inf, T_inc), 

                t_eval=np.arange(max_days))



## Plot result

plot_model_and_predict(train_loc, N, sol, title = 'SEIR Model (without intervention)')
# Define all variable of SEIR model 

T_inc = 5.2  # average incubation period

T_inf = 2.9  # average infectious period



# Define the intervention parameters (fit result, latter will show how to fit)

R_0, cfr, k, L=[ 3.95469597 , 0.04593316 , 3.      ,   15.32328881]



def time_varying_reproduction(t): 

    return R_0 / (1 + (t/L)**k)



sol2 = solve_ivp(SEIR_model, [0, max_days], [s, e, i, r], args=(time_varying_reproduction, T_inf, T_inc), 

                t_eval=np.arange(max_days))



plot_model_and_predict(train_loc, N, sol2, title = 'SEIR Model (with intervention)')
def cumsum_signal(vec):

    temp_val = 0

    vec_new = []

    for i in vec:

        if i > temp_val:

            vec_new.append(i)

            temp_val = i

        else:

            vec_new.append(temp_val)

    return vec_new
# Use a constant reproduction number

def eval_model_const(params, data, population, return_solution=False, forecast_days=0):

    R_0, cfr = params # Paramaters, R0 and cfr 

    N = population # Population of each country

    n_infected = data['ConfirmedCases'].iloc[0] # start from first comfirmedcase on dataset first date

    max_days = len(data) + forecast_days # How many days want to predict

    s, e, i, r = (N - n_infected)/ N, 0, n_infected / N, 0 #Initial stat for SEIR model

    

    # R0 become half after intervention days

    def time_varying_reproduction(t):

        if t > 60: # we set intervention days = 60

            return R_0 * 0.5

        else:

            return R_0

    

    # Solve the SEIR differential equation.

    sol = solve_ivp(SEIR_model, [0, max_days], [s, e, i, r], args=(time_varying_reproduction, T_inf, T_inc),

                    t_eval=np.arange(0, max_days))

    

    sus, exp, inf, rec = sol.y

    # Predict confirmedcase

    y_pred_cases = np.clip((inf + rec) * N ,0,np.inf)

    y_true_cases = data['ConfirmedCases'].values

    

    # Predict Fatalities by remove * fatality rate(cfr)

    y_pred_fat = np.clip(rec*N* cfr, 0, np.inf)

    y_true_fat = data['Fatalities'].values

    

    optim_days = min(20, len(data))  # Days to optimise for

    weights = 1 / np.arange(1, optim_days+1)[::-1]  # Recent data is more heavily weighted

    

    # using mean squre log error to evaluate

    msle_cases = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[-optim_days:], weights)

    msle_fat = mean_squared_log_error(y_true_fat[-optim_days:], y_pred_fat[-optim_days:], weights)

    msle_final = np.mean([msle_cases, msle_fat])

    

    if return_solution:

        return msle_final, sol

    else:

        return msle_final
# Use a Hill decayed reproduction number

def eval_model_decay(params, data, population, return_solution=False, forecast_days=0):

    R_0, cfr, k, L = params # Paramaters, R0 and cfr 

    N = population # Population of each country

    n_infected = data['ConfirmedCases'].iloc[0] # start from first comfirmedcase on dataset first date

    max_days = len(data) + forecast_days # How many days want to predict

    s, e, i, r = (N - n_infected)/ N, 0, n_infected / N, 0 #Initial stat for SEIR model

    

    # https://github.com/SwissTPH/openmalaria/wiki/ModelDecayFunctions   

    # Hill decay. Initial values: R_0=2.2, k=2, L=50

    def time_varying_reproduction(t): 

        return R_0 / (1 + (t/L)**k)

    

    # Solve the SEIR differential equation.

    sol = solve_ivp(SEIR_model, [0, max_days], [s, e, i, r], args=(time_varying_reproduction, T_inf, T_inc),

                    t_eval=np.arange(0, max_days))

    

    sus, exp, inf, rec = sol.y

    # Predict confirmedcase

    y_pred_cases = np.clip((inf + rec) * N ,0,np.inf)

    y_true_cases = data['ConfirmedCases'].values

    

    # Predict Fatalities by remove * fatality rate(cfr)

    y_pred_fat = np.clip(rec*N* cfr, 0, np.inf)

    y_true_fat = data['Fatalities'].values

    

    optim_days = min(20, len(data))  # Days to optimise for

    weights = 1 / np.arange(1, optim_days+1)[::-1]  # Recent data is more heavily weighted

    

    # using mean squre log error to evaluate

    msle_cases = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[-optim_days:], weights)

    msle_fat = mean_squared_log_error(y_true_fat[-optim_days:], y_pred_fat[-optim_days:], weights)

    msle_final = np.mean([msle_cases, msle_fat])

    

    if return_solution:

        return msle_final, sol

    else:

        return msle_final
from matplotlib import dates

import plotly.graph_objects as go



def fit_model_new(data, area_name, initial_guess=[2.2, 0.02, 2, 50], 

              bounds=((1, 20), (0, 0.15), (1, 3), (1, 100)), make_plot=True, decay_mode = None):

    

    if area_name in ['France']:# France last data looks weird, remove it

        train = data.query('ConfirmedCases > 0').copy()[:-1]

    else:

        train = data.query('ConfirmedCases > 0').copy()

    

    ####### Split Train & Valid #######

    valid_data = train[-7:]

    train_data = train[:-7]

    

    ####### If this country have no ConfirmedCase, return 0 #######

    if len(train_data) == 0:

        result_zero = np.zeros((43))

        return pd.DataFrame({'ConfirmedCases':result_zero,'Fatalities':result_zero}), 0 

    

    ####### Load the population of area #######

    try:

        #population = province_lookup[area_name]

        population = pop_info[pop_info['Name']==area_name]['Population'].tolist()[0]

    except IndexError:

        print ('country not in population set, '+str(area_name))

        population = 1000000 

    

    

    if area_name == 'US':

        population = 327200000

        

    cases_per_million = train_data['ConfirmedCases'].max() * 10**6 / population

    n_infected = train_data['ConfirmedCases'].iloc[0]

    

    ####### Total case/popuplation below 1, reduce country population #######

    if cases_per_million < 1:

        #print ('reduce pop divide by 100')

        population = population/100

        

    ####### Fit the real data by minimize the MSLE #######

    res_const = minimize(eval_model_const, [2.2, 0.02], bounds=((1, 20), (0, 0.15)),

                         args=(train_data, population, False),

                         method='L-BFGS-B')



    res_decay = minimize(eval_model_decay, initial_guess, bounds=bounds,

                         args=(train_data, population, False),

                         method='L-BFGS-B')

    

    ####### Align the date information #######

    test_end = datetime.strptime('2020-04-30','%Y-%m-%d')

    test_start = datetime.strptime('2020-03-19','%Y-%m-%d')

    test_period = (test_end - test_start).days

    train_max = train_data.Date.max()

    train_min = train_data.Date.min()

    add_date = 0

    delta_days =(test_end - train_max).days

    train_add_time=[]



    if train_min > test_start:

        add_date = (train_min-test_start).days

        last = train_min-pd.Timedelta(days=add_date)

        train_add_time = np.arange(last, train_min, dtype='datetime64[D]').tolist()

        train_add_time = pd.to_datetime(train_add_time)

        dates_all = train_add_time.append(pd.to_datetime(np.arange(train_min, test_end+pd.Timedelta(days=1), dtype='datetime64[D]')))

    else:

        dates_all = pd.to_datetime(np.arange(train_min, test_end+pd.Timedelta(days=1), dtype='datetime64[D]'))





    ####### Auto find the best decay function ####### 

    if decay_mode is None:

        if res_const.fun < res_decay.fun :

            msle, sol = eval_model_const(res_const.x, train_data, population, True, delta_days+add_date)

            res = res_const



        else:

            msle, sol = eval_model_decay(res_decay.x, train_data, population, True, delta_days+add_date)

            res = res_decay

            R_0, cfr, k, L = res.x

    else:

        if decay_mode =='day_decay':

            msle, sol = eval_model_const(res_const.x, train_data, population, True, delta_days+add_date)

            res = res_const

        else:

            msle, sol = eval_model_decay(res_decay.x, train_data, population, True, delta_days+add_date)

            res = res_decay

            R_0, cfr, k, L = res.x



    ####### Predict the result by using best fit paramater of SEIR model ####### 

    sus, exp, inf, rec = sol.y

    

    y_pred = pd.DataFrame({

        'ConfirmedCases': cumsum_signal(np.diff((inf + rec) * population, prepend=n_infected).cumsum()),

       # 'ConfirmedCases': [inf[0]*population for i in range(add_date)]+(np.clip((inf + rec) * population,0,np.inf)).tolist(),

       # 'Fatalities': [rec[0]*population for i in range(add_date)]+(np.clip(rec, 0, np.inf) * population * res.x[1]).tolist()

        'Fatalities': cumsum_signal((np.clip(rec * population * res.x[1], 0, np.inf)).tolist())

    })



    y_pred_valid = y_pred.iloc[len(train_data):len(train_data)+len(valid_data)]

    #y_pred_valid = y_pred.iloc[:len(train_data)]

    y_pred_test = y_pred.iloc[-(test_period+1):]

    #y_true_valid = train_data[['ConfirmedCases', 'Fatalities']]

    y_true_valid = valid_data[['ConfirmedCases', 'Fatalities']]

    #print (len(y_pred),train_min)

    #print (y_true_valid['ConfirmedCases'])

    #print (y_pred_valid['ConfirmedCases'])

    ####### Calculate MSLE ####### 

    valid_msle_cases = mean_squared_log_error(y_true_valid['ConfirmedCases'], y_pred_valid['ConfirmedCases'])

    valid_msle_fat = mean_squared_log_error(y_true_valid['Fatalities'], y_pred_valid['Fatalities'])

    valid_msle = np.mean([valid_msle_cases, valid_msle_fat])

    

    ####### Plot the fit result of train data and forecast after 250 days ####### 

    if make_plot:

        if len(res.x)<=2:

            print(f'Validation MSLE: {valid_msle:0.5f}, using intervention days decay, Reproduction number(R0) : {res.x[0]:0.5f}, Fatal rate : {res.x[1]:0.5f}')

        else:

            print(f'Validation MSLE: {valid_msle:0.5f}, using Hill decay, Reproduction number(R0) : {res.x[0]:0.5f}, Fatal rate : {res.x[1]:0.5f}, K : {res.x[2]:0.5f}, L: {res.x[3]:0.5f}')

        

        ####### Plot the fit result of train data dna SEIR model trends #######



        f = plt.figure(figsize=(16,5))

        ax = f.add_subplot(1,2,1)

        ax.plot(exp, 'y', label='Exposed');

        ax.plot(inf, 'r', label='Infected');

        ax.plot(rec, 'c', label='Recovered/deceased');

        plt.title('SEIR Model Trends')

        plt.xlabel("Days", fontsize=10);

        plt.ylabel("Fraction of population", fontsize=10);

        plt.legend(loc='best');

        #train_date_remove_year = train_data['Date'].apply(lambda date:'{:%m-%d}'.format(date))

        ax2 = f.add_subplot(1,2,2)

        xaxis = train_data['Date'].tolist()

        xaxis = dates.date2num(xaxis)

        hfmt = dates.DateFormatter('%m\n%d')

        ax2.xaxis.set_major_formatter(hfmt)

        ax2.plot(np.array(train_data['Date'], dtype='datetime64[D]'),train_data['ConfirmedCases'],label='Confirmed Cases (train)', c='g')

        ax2.plot(np.array(train_data['Date'], dtype='datetime64[D]'), y_pred['ConfirmedCases'][:len(train_data)],label='Cumulative modeled infections', c='r')

        ax2.plot(np.array(valid_data['Date'], dtype='datetime64[D]'), y_true_valid['ConfirmedCases'],label='Confirmed Cases (valid)', c='b')

        ax2.plot(np.array(valid_data['Date'], dtype='datetime64[D]'),y_pred_valid['ConfirmedCases'],label='Cumulative modeled infections (valid)', c='y')

        plt.title('Real ConfirmedCase and Predict ConfirmedCase')

        plt.legend(loc='best');

        plt.show()

            

        ####### Forecast 250 days after by using the best paramater of train data #######

        if len(res.x)>2:

            msle, sol = eval_model_decay(res.x, train_data, population, True, 250)

        else:

            msle, sol = eval_model_const(res.x, train_data, population, True, 250)

        

        sus, exp, inf, rec = sol.y

        

        y_pred = pd.DataFrame({

            'ConfirmedCases': cumsum_signal(np.diff((inf + rec) * population, prepend=n_infected).cumsum()),

            'Fatalities': cumsum_signal(np.clip(rec, 0, np.inf) * population * res.x[1])

        })

        

        ####### Plot 250 days after of each country #######

        start = train_min

        end = start + pd.Timedelta(days=len(y_pred))

        time_array = np.arange(start, end, dtype='datetime64[D]')



        max_day = numpy.where(inf == numpy.amax(inf))[0][0]

        where_time = time_array[max_day]

        pred_max_day = y_pred['ConfirmedCases'][max_day]

        xy_show_max_estimation = (where_time, max_day)

        

        con = y_pred['ConfirmedCases']

        max_day_con = numpy.where(con == numpy.amax(con))[0][0] # Find the max confimed case of each country

        max_con = numpy.amax(con)

        where_time_con = time_array[len(time_array)-50]

        xy_show_max_estimation_confirmed = (where_time_con, max_con)

        

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=time_array, y=y_pred['ConfirmedCases'].astype(int),

                            mode='lines',

                            line = dict(color='red'),

                            name='Estimation Confirmed Case Start from '+ str(start.date())+ ' to ' +str(end.date())))

        fig.add_trace(go.Scatter(x=time_array[:len(train)], y=train['ConfirmedCases'],

                            mode='lines',

                            name='Confirmed case until '+ str(train_max.date()),line = dict(color='green', width=4)))

        fig.add_annotation(

            x=where_time_con,

            y=max_con-(max_con/30),

            showarrow=False,

            text="Estimate Max Case around:" +str(int(max_con)),

            font=dict(

                color="Blue",

                size=15

            ))

        fig.add_annotation(

            x=time_array[len(train)-1],

            y=train['ConfirmedCases'].tolist()[-1],

            showarrow=True,

            text=f"Real Max ConfirmedCase: " +str(int(train['ConfirmedCases'].tolist()[-1]))) 

        

        fig.add_annotation(

            x=where_time,

            y=pred_max_day,

            text='Infect start decrease from: ' + str(where_time))   

        fig.update_layout(title='Estimate Confirmed Case ,'+area_name+' Total population ='+ str(int(population)), legend_orientation="h")

        fig.show()

        

        #df = pd.DataFrame({'Values': train_data['ConfirmedCases'].tolist()+y_pred['ConfirmedCases'].tolist(),'Date_datatime':time_array[:len(train_data)].tolist()+time_array.tolist(),

        #           'Real/Predict': ['ConfirmedCase' for i in range(len(train_data))]+['PredictCase' for i in range(len(y_pred))]})

        #fig = px.line(df, x="Date_datatime", y="Values",color = 'Real/Predict')

        #fig.show()

        #plt.figure(figsize = (16,7))

        #plt.plot(time_array[:len(train_data)],train_data['ConfirmedCases'],label='Confirmed case until '+ str(train_max.date()),color='g', linewidth=3.0)

        #plt.plot(time_array,y_pred['ConfirmedCases'],label='Estimation Confirmed Case Start from '+ str(start.date())+ ' to ' +str(end.date()),color='r', linewidth=1.0)

        #plt.annotate('Infect start decrease from: ' + str(where_time), xy=xy_show_max_estimation, size=15, color="black")

        #plt.annotate('max Confirmedcase: ' + str(int(max_con)), xy=xy_show_max_estimation_confirmed, size=15, color="black")

        #plt.title('Estimate Confirmed Case '+area_name+' Total population ='+ str(int(population)))

        #plt.legend(loc='lower right')

        #plt.show()





    return y_pred_test, valid_msle
country = 'Vietnam'

if country not in train['Country_Region'].unique():

    country_pd_train = train[train['Province_State']==country]

else:

    country_pd_train = train[train['Country_Region']==country]



a,b = fit_model_new(country_pd_train,country,make_plot=True)
country = 'US'

country_pd_train = train[train['Country_Region']==country]

country_pd_train2 = country_pd_train.groupby(['Date']).sum().reset_index()

country_pd_train2['Date'] = pd.to_datetime(country_pd_train2['Date'], format='%Y-%m-%d')

a,b = fit_model_new(country_pd_train2,country,make_plot=True)

country = 'California'

if country not in train['Country_Region'].unique():

    country_pd_train = train[train['Province_State']==country]

else:

    country_pd_train = train[train['Country_Region']==country]



a,b = fit_model_new(country_pd_train,country,make_plot=True)
country = 'New York'

if country not in train['Country_Region'].unique():

    country_pd_train = train[train['Province_State']==country]

else:

    country_pd_train = train[train['Country_Region']==country]



a,b = fit_model_new(country_pd_train,country,make_plot=True)
country = 'Italy'

if country not in train['Country_Region'].unique():

    country_pd_train = train[train['Province_State']==country]

else:

    country_pd_train = train[train['Country_Region']==country]



a,b = fit_model_new(country_pd_train,country,make_plot=True)
country = 'Spain'

if country not in train['Country_Region'].unique():

    country_pd_train = train[train['Province_State']==country]

else:

    country_pd_train = train[train['Country_Region']==country]



a,b = fit_model_new(country_pd_train,country,make_plot=True)
country = 'Germany'

if country not in train['Country_Region'].unique():

    country_pd_train = train[train['Province_State']==country]

else:

    country_pd_train = train[train['Country_Region']==country]



a,b = fit_model_new(country_pd_train,country,make_plot=True)
validation_scores = []

validation_county = []

validation_country = []



test_seir = test.copy()



for country in tqdm(train['Country_Region'].unique()):

    country_pd_train = train[train['Country_Region']==country]

    #if country_pd_train['Province_State'].isna().unique()==True:

    if len(country_pd_train['Province_State'].unique())<2:

        predict_test, score = fit_model_new(country_pd_train,country,make_plot=False)

        if score ==0:

            print(f'{country} no case')

        validation_scores.append(score)

        validation_county.append(country)

        validation_country.append(country)

        test_seir.loc[test_seir['Country_Region']==country,'ConfirmedCases'] = predict_test['ConfirmedCases'].tolist()

        test_seir.loc[test_seir['Country_Region']==country,'Fatalities'] = predict_test['Fatalities'].tolist()

    else:

        for state in country_pd_train['Province_State'].unique():

            if state != state: # check nan

                state_pd = country_pd_train[country_pd_train['Province_State'].isna()]

                predict_test, score = fit_model_new(state_pd,state,make_plot=False)

                if score ==0:

                    print(f'{country} / {state} no case')

                validation_scores.append(score)

                validation_county.append(state)

                validation_country.append(country)

                test_seir.loc[(test_seir['Country_Region']==country)&(test_seir['Province_State'].isna()),'ConfirmedCases'] = predict_test['ConfirmedCases'].tolist()

                test_seir.loc[(test_seir['Country_Region']==country)&(test_seir['Province_State'].isna()),'Fatalities'] = predict_test['Fatalities'].tolist()

            else:

                state_pd = country_pd_train[country_pd_train['Province_State']==state]

                predict_test, score = fit_model_new(state_pd,state,make_plot=False)

                if score ==0:

                    print(f'{country} / {state} no case')

                validation_scores.append(score)

                validation_county.append(state)

                validation_country.append(country)

                test_seir.loc[(test_seir['Country_Region']==country)&(test_seir['Province_State']==state),'ConfirmedCases'] = predict_test['ConfirmedCases'].tolist()

                test_seir.loc[(test_seir['Country_Region']==country)&(test_seir['Province_State']==state),'Fatalities'] = predict_test['Fatalities'].tolist()

         #   print(f'{country} {state} {score:0.5f}')

            

print(f'Mean validation score: {np.average(validation_scores):0.5f}')
validation_scores = pd.DataFrame({'country/state':validation_country,'country':validation_county,'MSLE':validation_scores})

validation_scores.sort_values(by=['MSLE'], ascending=False).head(20)
large_msle = validation_scores[validation_scores['MSLE']>1]
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression



for country in large_msle['country'].unique():

    if (country!= country)==False: # check None

        #print ('training model for country ==>'+country)

        country_pd_train = train[train['Country_Region']==country]

        country_pd_test = test[test['Country_Region']==country]

        if len(country_pd_train)==0:

            country_pd_train = train[train['Province_State']==country]

            country_pd_test = test[test['Province_State']==country]



            x = np.array(range(len(country_pd_train))).reshape((-1,1))[:-7]

            valid_x = np.array(range(len(country_pd_train))).reshape((-1,1))[-7:]

            y = country_pd_train['ConfirmedCases'][:-7]

            valid_y = country_pd_train['ConfirmedCases'][-7:]

            y_fat = country_pd_train['Fatalities'][:-7]

            valid_y_fat = country_pd_train['Fatalities'][-7:]

            

            model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                             ('linear', LinearRegression(fit_intercept=False))])

            model = model.fit(x, y)



            model_fat = Pipeline([('poly', PolynomialFeatures(degree=2)),

                             ('linear', LinearRegression(fit_intercept=False))])

            model_fat = model_fat.fit(x, y_fat)

            

            predict_y = model.predict(valid_x)

            predict_yfat = model_fat.predict(valid_x)

            score = mean_squared_log_error(np.clip(valid_y,0,np.inf), np.clip(predict_y,0,np.inf))

            score_fat = mean_squared_log_error(np.clip(valid_y_fat,0,np.inf), np.clip(predict_yfat,0,np.inf))

            score = (score+score_fat)/2



            print(f'{country} {score:0.5f}')

            if score < large_msle[large_msle['country']==country]['MSLE'].tolist()[0]:

                validation_scores.loc[validation_scores['country']==country,'MSLE'] = score

                predict_x = (np.array(range(len(country_pd_test)))+50).reshape((-1,1))

                test_seir.loc[test_seir['Province_State']==country,'ConfirmedCases'] = model.predict(predict_x)

                test_seir.loc[test_seir['Province_State']==country,'Fatalities'] = model_fat.predict(predict_x)

        else:

            x = np.array(range(len(country_pd_train))).reshape((-1,1))[:-7]

            valid_x = np.array(range(len(country_pd_train))).reshape((-1,1))[-7:]

            y = country_pd_train['ConfirmedCases'][:-7]

            valid_y = country_pd_train['ConfirmedCases'][-7:]

            y_fat = country_pd_train['Fatalities'][:-7]

            valid_y_fat = country_pd_train['Fatalities'][-7:]

            

            model = Pipeline([('poly', PolynomialFeatures(degree=2)),

                             ('linear', LinearRegression(fit_intercept=False))])

            model = model.fit(x, y)



            model_fat = Pipeline([('poly', PolynomialFeatures(degree=2)),

                             ('linear', LinearRegression(fit_intercept=False))])

            model_fat = model_fat.fit(x, y_fat)

            

            predict_y = model.predict(valid_x)

            predict_yfat = model_fat.predict(valid_x)

            score = mean_squared_log_error(np.clip(valid_y,0,np.inf), np.clip(predict_y,0,np.inf))

            score_fat = mean_squared_log_error(np.clip(valid_y_fat,0,np.inf), np.clip(predict_yfat,0,np.inf))

            score = (score+score_fat)/2



            print(f'{country} {score:0.5f}')

            if score < large_msle[large_msle['country']==country]['MSLE'].tolist()[0]:

                validation_scores.loc[validation_scores['country']==country,'MSLE'] = score

                predict_x = (np.array(range(len(country_pd_test)))+50).reshape((-1,1))

                test_seir.loc[test_seir['Country_Region']==country,'ConfirmedCases'] = model.predict(predict_x)

                test_seir.loc[test_seir['Country_Region']==country,'Fatalities'] = model_fat.predict(predict_x)

                
val_soces = validation_scores['MSLE'].tolist()

print(f'Mean validation score: {np.average(val_soces):0.5f}')


submit['Fatalities'] = round(test_seir['Fatalities'].astype('float'),0)

submit['ConfirmedCases'] = round(test_seir['ConfirmedCases'].astype('float'),0)

submit.tail()

submit.to_csv('submission.csv',index=False)
train_rfm = train.copy()



# Add additional variables

#month

train_rfm['month'] = train_rfm['Date'].dt.month



#date

train_rfm['dates'] = train_rfm['Date'].dt.day





### do the same for test data

test_rfm = test.copy()



# Add additional variables



#month

test_rfm['month'] = test_rfm['Date'].dt.month



#date

test_rfm['dates'] = test_rfm['Date'].dt.day





train_rfm.tail()

countries_array = train['Country_Region'].unique()



train_first_result = pd.DataFrame()



for i in countries_array:

    # get relevant data 

    day_first_outbreak = train_rfm.loc[train_rfm['Country_Region']==i]

    

    date_outbreak = day_first_outbreak.loc[day_first_outbreak['ConfirmedCases']>0]['Date'].min()

    

    #Calculate days since first outbreak happened

    day_first_outbreak['days_since_first_outbreak'] = (day_first_outbreak['Date'] - date_outbreak).astype('timedelta64[D]')



    

    #impute the negative days with 0

    day_first_outbreak['days_since_first_outbreak'][day_first_outbreak['days_since_first_outbreak']<0] = 0 

   

    train_first_result = train_first_result.append(day_first_outbreak,ignore_index=True)





### do the same for test data



test_first_result = pd.DataFrame()



for i in countries_array:

    # get relevant data 

    day_first_outbreak = test_rfm.loc[test_rfm['Country_Region']==i]

    

    day_first_outbreak_train = train_rfm.loc[train_rfm['Country_Region']==i]

    

    date_outbreak = day_first_outbreak_train.loc[day_first_outbreak_train['ConfirmedCases']>0]['Date'].min()

    

    #Calculate days since first outbreak happened

    day_first_outbreak['days_since_first_outbreak'] = (day_first_outbreak['Date'] - date_outbreak).astype('timedelta64[D]')



    

    #impute the negative days with 0

    day_first_outbreak['days_since_first_outbreak'][day_first_outbreak['days_since_first_outbreak']<0] = 0 

   

    test_first_result = test_first_result.append(day_first_outbreak,ignore_index=True)





test_first_result.tail()

#ecoding countries data



#train

labels, values = pd.factorize(train_first_result['Country_Region'])



train_first_result['country_id'] = labels



train_first_result.head()



#test



labels, values = pd.factorize(test_first_result['Country_Region'])



test_first_result['country_id'] = labels



test_first_result.head()
random.seed(123)



X = train_first_result[['country_id','month','dates','days_since_first_outbreak']]



y_confirm = train_first_result['ConfirmedCases']

y_fatal = train_first_result['Fatalities']



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_confirm, test_size=0.3, random_state=0)





#RF model

rf = RandomForestClassifier()



predict_labels = X.columns



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")



# Print the name and gini importance of each feature

for feature in zip(predict_labels, rf.feature_importances_):

    print(feature)

    
sns.set(rc={'figure.figsize':(15, 7)})



plt.figure()



plt.title("Feature importances",fontsize=20)



plt.bar(predict_labels,rf.feature_importances_, align="center", color='dodgerblue')



plt.xticks(predict_labels)



plt.xticks(rotation=90)



plt.show()


## predict on test set



confirm_case = rf.predict(test_first_result[['country_id','month','dates','days_since_first_outbreak']])



random.seed(123)



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_fatal, test_size=0.3, random_state=0)



#RF model

rf = RandomForestClassifier()



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")



## predict on test set



fatal_case = rf.predict(test_first_result[['country_id','month','dates','days_since_first_outbreak']])



forecaseId = pd.DataFrame(test[['ForecastId']])

confirm_case = pd.DataFrame(confirm_case)

fatal_case = pd.DataFrame(fatal_case)





final_result_rf = pd.concat([forecaseId,confirm_case,fatal_case],axis=1)

final_result_rf.columns = ['ForecastId','ConfirmedCases','Fatalities']



final_result_rf.tail()
final_result_rf.to_csv('results_rfm.csv', index = False)
# function that apply Random Forest for each country in the data



def rf_each_country(seed, original_df, train_df, test_df, confirm , 

                    fatal, xlabels, test_size = 0.3):



    random.seed(seed)



    countries_array = original_df['Country_Region'].unique()



    final_result = pd.DataFrame()

    

    confirm_test_case = pd.DataFrame()

    fatal_test_case = pd.DataFrame()

    

    confirm_val = pd.DataFrame()

    fatal_val = pd.DataFrame()



    ##predict confirmed cases

    for i in countries_array:

        # get relevant data 

        train_set = train_df.loc[train_df['Country_Region']==i]

        test_set = test_df.loc[test_df['Country_Region']==i]





        #Confirm case

        X1 = train_set[xlabels]

        y1 = train_set[confirm]



        #train test split

        X_train_rf1, X_test_rf1, y_train_rf1, y_test_rf1 = train_test_split(X1, y1, test_size=0.3, random_state=0)



        #RF model

        rf1 = RandomForestClassifier()



        # Train the classifier

        rf1.fit(X_train_rf1, y_train_rf1)

        

        test_confirm = rf1.predict(X_test_rf1)



        ## predict on test set



        confirm_case = rf1.predict(test_set[xlabels])





        #Fatal case

        X2 = train_set[xlabels]

        y2 = train_set[fatal]



        #train test split

        X_train_rf2, X_test_rf2, y_train_rf2, y_test_rf2 = train_test_split(X2, y2, test_size=0.3, random_state=0)



        #RF model

        rf2 = RandomForestClassifier()



        # Train the classifier

        rf2.fit(X_train_rf2, y_train_rf2)

        

        test_fatal = rf2.predict(X_test_rf2)



        ## predict on test set



        fatal_case = rf2.predict(test_set[xlabels])

        

        

        ## combine them together and meausre RMSE

        test_confirm = pd.DataFrame(test_confirm)

        test_fatal = pd.DataFrame(test_fatal)

        

        y_test_rf1 = pd.DataFrame(y_test_rf1)

        y_test_rf2 = pd.DataFrame(y_test_rf2)

        

        

        confirm_test_case = confirm_test_case.append(test_confirm)

        fatal_test_case = fatal_test_case.append(test_fatal)

        

        confirm_val = confirm_val.append(y_test_rf1)

        fatal_val = fatal_val.append(y_test_rf2)

        

        

        ### Combine results

        

        confirm_case = pd.DataFrame(confirm_case)

        fatal_case = pd.DataFrame(fatal_case)



        final_result_pred = pd.concat([confirm_case,fatal_case],axis=1)



        final_result_pred.columns = ['ConfirmedCases','Fatalities']

        

        final_result = final_result.append(final_result_pred, ignore_index=True)

    

    

    ##Print out validation metrics

    

    confirm_test_case = np.array(confirm_test_case)

    fatal_test_case = np.array(fatal_test_case)

    

    

    confirm_va1 = np.array(confirm_val)

    fatal_val1 = np.array(fatal_val)

    



    print('Mean Absolute Error for Confirmed Case Prediction:', round(metrics.mean_absolute_error(confirm_va1, confirm_test_case),2))  

    print('Mean Squared Error for Confirmed Case Prediction:', round(metrics.mean_squared_error(confirm_va1, confirm_test_case),2))  

    print('Root Mean Squared Error for Confirmed Case Prediction:', round(np.sqrt(metrics.mean_squared_error(confirm_va1, confirm_test_case)),2), "\n")





    print('Mean Absolute Error for Fatal Case Prediction:', round(metrics.mean_absolute_error(fatal_val1, fatal_test_case),2))  

    print('Mean Squared Error for Fatal Case Prediction:', round(metrics.mean_squared_error(fatal_val1, fatal_test_case),2))  

    print('Root Mean Squared Error for Fatal Case Prediction:', round(np.sqrt(metrics.mean_squared_error(fatal_val1, fatal_test_case)),2), "\n")



    

    #compile with prediction IDs

    

    forecaseId = pd.DataFrame(test[['ForecastId']])



    final_result = pd.concat([forecaseId,final_result],axis=1)



    return final_result



x_pred_lab = ['month','dates','days_since_first_outbreak']



final_result_rf_each = rf_each_country(seed=123, original_df=train, train_df=train_first_result, 

                                       test_df=test_first_result, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)



final_result_rf_each.tail()
final_result_rf_each.to_csv('result_rf_each.csv',index=False)
#construct the OLS model

X = train_first_result[['country_id','month','dates','days_since_first_outbreak']]

y_confirm = train_first_result['ConfirmedCases']

y_fatal = train_first_result['Fatalities']



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train, X_test, y_train, y_test = train_test_split(X, y_confirm, test_size=0.3, random_state=0)



X_train  = np.array(X_train)

X_test = np.array(X_test)

y_train  = np.array(y_train)

y_test  = np.array(y_test)



# Note the difference in argument order

model = LinearRegression()



model.fit(X_train,y_train)



predictions = model.predict(X_test) # make the predictions by the model



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, predictions),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, predictions),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2))

    


confirm_case = model.predict(

    test_first_result[['country_id','month','dates','days_since_first_outbreak']]) # make the predictions by the model

#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_fatal, test_size=0.3, random_state=0)



X_train  = np.array(X_train)

X_test = np.array(X_test)

y_train  = np.array(y_train)

y_test  = np.array(y_test)



# Note the difference in argument order

model = LinearRegression()



model.fit(X_train,y_train)



predictions = model.predict(X_test) # make the predictions by the model



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, predictions),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, predictions),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2))

   


fatal_case = model.predict(

    test_first_result[['country_id','month','dates','days_since_first_outbreak']]) # make the predictions by the model

forecaseId = pd.DataFrame(test[['ForecastId']])

confirm_case = round(pd.DataFrame(confirm_case),0)

fatal_case = round(pd.DataFrame(fatal_case),0)





final_result_lin = pd.concat([forecaseId,confirm_case,fatal_case],axis=1)

final_result_lin.columns = ['ForecastId','ConfirmedCases','Fatalities']



final_result_lin.tail()
final_result_lin.to_csv('results_lin.csv',index=False)
# function that apply Linear Regression for each country in the data



def lin_each_country(seed, original_df, train_df, test_df, confirm , 

                    fatal, xlabels, test_size = 0.3):



    random.seed(seed)



    countries_array = original_df['Country_Region'].unique()



    final_result = pd.DataFrame()

    

    confirm_test_case = pd.DataFrame()

    fatal_test_case = pd.DataFrame()

    

    confirm_val = pd.DataFrame()

    fatal_val = pd.DataFrame()



    ##predict confirmed cases

    for i in countries_array:

        # get relevant data 

        train_set = train_df.loc[train_df['Country_Region']==i]

        test_set = test_df.loc[test_df['Country_Region']==i]





        #Confirm case

        X1 = train_set[xlabels]

        y1 = train_set[confirm]



        #train test split

        X_train_rf1, X_test_rf1, y_train_rf1, y_test_rf1 = train_test_split(X1, y1, test_size=0.3, random_state=0)



        X_train_rf1  = np.array(X_train_rf1)

        X_test_rf1 = np.array(X_test_rf1)

        y_train_rf1  = np.array(y_train_rf1)

        y_test_rf1  = np.array(y_test_rf1)



        # Note the difference in argument order

        model = LinearRegression()



        model.fit(X_train_rf1,y_train_rf1)



        test_confirm = model.predict(X_test_rf1) # make the predictions by the model



        

        ## predict on test set



        confirm_case = model.predict(test_set[xlabels])





        #Fatal case

        X2 = train_set[xlabels]

        y2 = train_set[fatal]



        #train test split

        X_train_rf2, X_test_rf2, y_train_rf2, y_test_rf2 = train_test_split(X2, y2, test_size=0.3, random_state=0)



        X_train_rf2  = np.array(X_train_rf2)

        X_test_rf2 = np.array(X_test_rf2)

        y_train_rf2  = np.array(y_train_rf2)

        y_test_rf2  = np.array(y_test_rf2)



        # Note the difference in argument order

        model = LinearRegression()



        model.fit(X_train_rf2,y_train_rf2)



        test_fatal = model.predict(X_test_rf2) # make the predictions by the model



        

        ## predict on test set



        fatal_case = model.predict(test_set[xlabels])

        

        

        ## combine them together and meausre RMSE

        test_confirm = pd.DataFrame(test_confirm)

        test_fatal = pd.DataFrame(test_fatal)

        

        y_test_rf1 = pd.DataFrame(y_test_rf1)

        y_test_rf2 = pd.DataFrame(y_test_rf2)

        

        

        confirm_test_case = confirm_test_case.append(test_confirm)

        fatal_test_case = fatal_test_case.append(test_fatal)

        

        confirm_val = confirm_val.append(y_test_rf1)

        fatal_val = fatal_val.append(y_test_rf2)

        

        

        ### Combine results

        

        confirm_case = round(pd.DataFrame(confirm_case),0)

        fatal_case = round(pd.DataFrame(fatal_case),0)



        final_result_pred = pd.concat([confirm_case,fatal_case],axis=1)



        final_result_pred.columns = ['ConfirmedCases','Fatalities']

        

        final_result = final_result.append(final_result_pred, ignore_index=True)

    

    

    ##Print out validation metrics

    

    confirm_test_case = np.array(confirm_test_case)

    fatal_test_case = np.array(fatal_test_case)

    

    

    confirm_va1 = np.array(confirm_val)

    fatal_val1 = np.array(fatal_val)

    



    print('Mean Absolute Error for Confirmed Case Prediction:', round(metrics.mean_absolute_error(confirm_va1, confirm_test_case),2))  

    print('Mean Squared Error for Confirmed Case Prediction:', round(metrics.mean_squared_error(confirm_va1, confirm_test_case),2))  

    print('Root Mean Squared Error for Confirmed Case Prediction:', round(np.sqrt(metrics.mean_squared_error(confirm_va1, confirm_test_case)),2), "\n")





    print('Mean Absolute Error for Fatal Case Prediction:', round(metrics.mean_absolute_error(fatal_val1, fatal_test_case),2))  

    print('Mean Squared Error for Fatal Case Prediction:', round(metrics.mean_squared_error(fatal_val1, fatal_test_case),2))  

    print('Root Mean Squared Error for Fatal Case Prediction:', round(np.sqrt(metrics.mean_squared_error(fatal_val1, fatal_test_case)),2), "\n")



    

    #compile with prediction IDs

    

    forecaseId = pd.DataFrame(test[['ForecastId']])



    final_result = pd.concat([forecaseId,final_result],axis=1)



    return final_result

x_pred_lab = ['month','dates','days_since_first_outbreak']



final_result_lin_each = lin_each_country(seed=123, original_df=train, train_df=train_first_result, 

                                       test_df=test_first_result, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)





final_result_lin_each.tail()
final_result_lin_each.to_csv("result_lin_each.csv",index=False)


X = train_first_result[['country_id','month','dates','days_since_first_outbreak']]

y_confirm = train_first_result['ConfirmedCases']

y_fatal = train_first_result['Fatalities']





#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_br, X_test_br, y_train_br, y_test_br = train_test_split(X, y_confirm, test_size=0.3, random_state=0)



predict_labels = X.columns



clf = BayesianRidge(compute_score=True)

clf.fit(X_train_br, y_train_br)



y_pred_br = clf.predict(X_test_br)





print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_br, y_pred_br),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_br, y_pred_br),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_br, y_pred_br)),2))

## predict on test set



confirm_case = clf.predict(test_first_result[['country_id','month','dates','days_since_first_outbreak']])





#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_br, X_test_br, y_train_br, y_test_br = train_test_split(X, y_fatal, test_size=0.3, random_state=0)



predict_labels = X.columns



clf = BayesianRidge(compute_score=True)

clf.fit(X_train_br, y_train_br)



y_pred_br = clf.predict(X_test_br)



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_br, y_pred_br),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_br, y_pred_br),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_br, y_pred_br)),2))



## predict on test set



fatal_case = clf.predict(test_first_result[['country_id','month','dates','days_since_first_outbreak']])



forecaseId = pd.DataFrame(test[['ForecastId']])

confirm_case = round(pd.DataFrame(confirm_case),0)

fatal_case = round(pd.DataFrame(fatal_case),0)





final_result_br = pd.concat([forecaseId,confirm_case,fatal_case],axis=1)

final_result_br.columns = ['ForecastId','ConfirmedCases','Fatalities']



final_result_br.tail()

final_result_br.to_csv('results_br.csv', index = False)
# function that apply Bayesian Regression for each country in the data



def br_each_country(seed, original_df, train_df, test_df, confirm , 

                    fatal, xlabels, test_size = 0.3):



    random.seed(seed)



    countries_array = original_df['Country_Region'].unique()



    final_result = pd.DataFrame()

    

    confirm_test_case = pd.DataFrame()

    fatal_test_case = pd.DataFrame()

    

    confirm_val = pd.DataFrame()

    fatal_val = pd.DataFrame()



    ##predict confirmed cases

    for i in countries_array:

        # get relevant data 

        train_set = train_df.loc[train_df['Country_Region']==i]

        test_set = test_df.loc[test_df['Country_Region']==i]





        #Confirm case

        X1 = train_set[xlabels]

        y1 = train_set[confirm]



        #train test split

        X_train_rf1, X_test_rf1, y_train_rf1, y_test_rf1 = train_test_split(X1, y1, test_size=0.3, random_state=0)



        #Bayesian Ridge Model

        clf = BayesianRidge(compute_score=True)

        

        ##train

        clf.fit(X_train_rf1, y_train_rf1)



        test_confirm = clf.predict(X_test_rf1)

  

        ## predict on test set



        confirm_case = clf.predict(test_set[xlabels])





        #Fatal case

        X2 = train_set[xlabels]

        y2 = train_set[fatal]



        #train test split

        X_train_rf2, X_test_rf2, y_train_rf2, y_test_rf2 = train_test_split(X2, y2, test_size=0.3, random_state=0)

        

        #Bayesian Ridge Model

        clf = BayesianRidge(compute_score=True)

        

        ##train

        clf.fit(X_train_rf2, y_train_rf2)



        test_fatal = clf.predict(X_test_rf2)



        ## predict on test set



        fatal_case = clf.predict(test_set[xlabels])

        

        

        ## combine them together and meausre RMSE

        test_confirm = pd.DataFrame(test_confirm)

        test_fatal = pd.DataFrame(test_fatal)

        

        y_test_rf1 = pd.DataFrame(y_test_rf1)

        y_test_rf2 = pd.DataFrame(y_test_rf2)

        

        

        confirm_test_case = confirm_test_case.append(test_confirm)

        fatal_test_case = fatal_test_case.append(test_fatal)

        

        confirm_val = confirm_val.append(y_test_rf1)

        fatal_val = fatal_val.append(y_test_rf2)

        

        

        ### Combine results

        

        confirm_case = round(pd.DataFrame(confirm_case),0)

        fatal_case = round(pd.DataFrame(fatal_case),0)



        final_result_pred = pd.concat([confirm_case,fatal_case],axis=1)



        final_result_pred.columns = ['ConfirmedCases','Fatalities']

        

        final_result = final_result.append(final_result_pred, ignore_index=True)

    

    

    ##Print out validation metrics

    

    confirm_test_case = np.array(confirm_test_case)

    fatal_test_case = np.array(fatal_test_case)

    

    

    confirm_va1 = np.array(confirm_val)

    fatal_val1 = np.array(fatal_val)

    



    print('Mean Absolute Error for Confirmed Case Prediction:', round(metrics.mean_absolute_error(confirm_va1, confirm_test_case),2))  

    print('Mean Squared Error for Confirmed Case Prediction:', round(metrics.mean_squared_error(confirm_va1, confirm_test_case),2))  

    print('Root Mean Squared Error for Confirmed Case Prediction:', round(np.sqrt(metrics.mean_squared_error(confirm_va1, confirm_test_case)),2), "\n")





    print('Mean Absolute Error for Fatal Case Prediction:', round(metrics.mean_absolute_error(fatal_val1, fatal_test_case),2))  

    print('Mean Squared Error for Fatal Case Prediction:', round(metrics.mean_squared_error(fatal_val1, fatal_test_case),2))  

    print('Root Mean Squared Error for Fatal Case Prediction:', round(np.sqrt(metrics.mean_squared_error(fatal_val1, fatal_test_case)),2), "\n")



    

    #compile with prediction IDs

    

    forecaseId = pd.DataFrame(test[['ForecastId']])



    final_result = pd.concat([forecaseId,final_result],axis=1)



    return final_result

x_pred_lab = ['month','dates','days_since_first_outbreak']



final_result_br_each = br_each_country(seed=123, original_df=train, train_df=train_first_result, 

                                       test_df=test_first_result, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)





final_result_br_each.tail()
final_result_br_each.to_csv("result_br_each.csv",index=False)
weather_train = weather_data.copy()



# Add additional variables

#month

weather_train['month'] = weather_train['Date'].dt.month



#date

weather_train['dates'] = weather_train['Date'].dt.day





### do the same for test data

weather_test1 = weather_test.copy()



# Add additional variables



#month

weather_test1['month'] = weather_test1['Date'].dt.month



#date

weather_test1['dates'] = weather_test1['Date'].dt.day





weather_test1.tail()
#ecoding countries data



#train

labels, values = pd.factorize(weather_train['Country_Region'])



weather_train['country_id'] = labels



weather_train.head()



#test



labels, values = pd.factorize(weather_test1['Country_Region'])



weather_test1['country_id'] = labels



weather_test1.head()
random.seed(123)



X = weather_train[['country_id','Lat', 'Long','day_from_jan_first', 'temp', 'stp', 'wdsp', 'prcp','fog', 'month', 'dates']]



y_confirm = weather_train['ConfirmedCases']



y_fatal = weather_train['Fatalities']





#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_confirm, test_size=0.3, random_state=0)



predict_labels = X.columns



#RF model

rf = RandomForestClassifier()



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)





print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")





# Print the name and gini importance of each feature

for feature in zip(predict_labels, rf.feature_importances_):

    print(feature)

    
plt.figure()



plt.title("Feature importances",fontsize=20)



plt.bar(predict_labels,rf.feature_importances_, align="center", color='dodgerblue')



plt.xticks(predict_labels)



plt.xticks(rotation=90)



plt.show()


## predict on test set



confirm_case = rf.predict(weather_test1[['country_id','Lat', 'Long','day_from_jan_first', 'temp',

                                         'stp', 'wdsp', 'prcp','fog','month', 'dates']])

random.seed(123)



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_fatal, test_size=0.3, random_state=0)



predict_labels = X.columns



#RF model

rf = RandomForestClassifier()



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")



## predict on test set



fatal_case = rf.predict(weather_test1[['country_id','Lat', 'Long','day_from_jan_first', 'temp',

                                         'stp', 'wdsp', 'prcp','fog', 'month', 'dates']])



forecaseId = pd.DataFrame(test[['ForecastId']])

confirm_case = pd.DataFrame(confirm_case)

fatal_case = pd.DataFrame(fatal_case)





final_result_rf = pd.concat([forecaseId,confirm_case,fatal_case],axis=1)

final_result_rf.columns = ['ForecastId','ConfirmedCases','Fatalities']



final_result_rf.tail()

final_result_rf.to_csv('results_rf_weather.csv', index = False)
x_pred_lab = ['Lat', 'Long','day_from_jan_first', 'temp', 'stp', 'wdsp', 'prcp','fog','month', 'dates']



final_result_rf_weather_each = rf_each_country(seed=123, original_df=weather_train, train_df=weather_train, test_df=weather_test1, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_rf_weather_each.tail()
final_result_rf_weather_each.to_csv('submission.csv',index=False)
#run for each country

x_pred_lab = ['Lat', 'Long','day_from_jan_first', 'temp', 'stp', 'wdsp', 'prcp','fog','month', 'dates']



final_result_lin_weather_each = lin_each_country(seed=123, original_df=weather_train, train_df=weather_train, test_df=weather_test1, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_lin_weather_each.tail()
final_result_lin_weather_each.to_csv('final_result_lin_weather_each.csv',index=False)
#run for each country

x_pred_lab = ['Lat', 'Long','day_from_jan_first', 'temp', 'stp', 'wdsp', 

                   'prcp','fog', 'month', 'dates']



final_result_br_weather_each = br_each_country(seed=123, original_df=weather_train, train_df=weather_train, test_df=weather_test1, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_br_weather_each.tail()
final_result_br_weather_each.to_csv('final_result_br_weather_each.csv',index=False)
import math

from math import radians, cos, sin, asin, sqrt
def distance(org_lat,org_lon, dest_lat, dest_lon):

    

    r = 3959 # miles

    # for km, we use r = 6371

    

    org_lat, org_lon, dest_lat, dest_lon = map(radians,[org_lat, org_lon, dest_lat, dest_lon])

    dlon = dest_lon - org_lon 

    dlat = dest_lat - org_lat 

    

    a = sin(dlat/2)**2 + cos(org_lat) * cos(dest_lat) * sin(dlon/2)**2

    

    c = 2 * asin(sqrt(a)) 

    

    return c * r



complete_distance = complete_data.copy()



complete_distance['france_lat'] = 46.2276

complete_distance['france_long'] = 2.2137

complete_distance['us_lat'] = 37.0902

complete_distance['us_long'] = -95.7129

complete_distance['china_lat'] = 30.5928

complete_distance['china_long'] = 114.3055





complete_distance.head()

complete_distance = complete_distance.reset_index()



europe = ['Austria','Italy','Belgium','Latvia','Bulgaria','Lithuania','Croatia','Luxembourg',

          'Cyprus','Malta','Czechia','Netherlands','Denmark','Poland','Estonia','Portugal',

          'Finland','Romania','France','Slovakia','Germany','Slovenia','Greece','Spain',

          'Hungary','Sweden','Ireland','Switzerland','United Kingdom']



europe_dis = complete_distance.loc[complete_distance['Country_Region'].isin(europe)==True]



north_america = ['Antigua and Barbuda','Bahamas','Barbados','Belize','Canada','Costa Rica','Cuba','El Salvador',

                 'Grenada','Guatemala','Haití','Honduras','Jamaica','Mexico','Nicaragua','Panama',

                 'Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines','Trinidad and Tobago','US']





america_dis = complete_distance.loc[complete_distance['Country_Region'].isin(north_america)==True]



asia_dis = complete_distance.loc[complete_distance['Country_Region'].isin(europe)==False]

asia_dis = asia_dis.loc[asia_dis['Country_Region'].isin(north_america)==False]

#Calculate distance to Europe areas



europe_dis['distance_to_first_outbreak'] = europe_dis.apply(lambda x: distance(x['Lat'],x['Long'],x['france_lat'],

                                                                 x['france_long']), axis=1)



europe_dis.head()

#Calculate distance to America areas



america_dis['distance_to_first_outbreak'] = america_dis.apply(lambda x: distance(x['Lat'],x['Long'],x['us_lat'],

                                                                 x['us_long']), axis=1)



america_dis.head()

#Calculate distance to China for the rest of the countries



asia_dis['distance_to_first_outbreak'] = asia_dis.apply(lambda x: distance(x['Lat'],x['Long'],x['china_lat'],

                                                                 x['china_long']), axis=1)



asia_dis.head()



complete_distance1 = pd.DataFrame()



complete_distance1 = complete_distance1.append(europe_dis)

complete_distance1 = complete_distance1.append(america_dis)

complete_distance1 = complete_distance1.append(asia_dis)



complete_distance1 = complete_distance1.sort_values('index')



complete_distance1 = complete_distance1.set_index('index')



complete_distance1.head()

train_dis = pd.merge(train_rfm,

                 complete_distance1[['Country_Region','distance_to_first_outbreak']],

                 on=['Country_Region'], 

                 how='left')



train_dis['distance_to_first_outbreak'] = train_dis['distance_to_first_outbreak'].interpolate(

    method ='linear', limit_direction ='both') 



train_dis = train_dis.drop_duplicates(['Id'],keep='first')





test_dis = pd.merge(test_rfm,

                 complete_distance1[['Country_Region','distance_to_first_outbreak']],

                 on=['Country_Region'], 

                 how='left')





test_dis['distance_to_first_outbreak'] = test_dis['distance_to_first_outbreak'].interpolate(

    method ='linear', limit_direction ='both') 



test_dis = test_dis.drop_duplicates(['ForecastId'],keep='first')

#ecoding countries data

labels, values = pd.factorize(train_dis['Country_Region'])



train_dis['country_id'] = labels



train_dis.head()



labels, values = pd.factorize(test_dis['Country_Region'])



test_dis['country_id'] = labels



test_dis.head()

random.seed(123)



X = train_dis[['country_id','month','dates','distance_to_first_outbreak']]

y_confirm = train_dis['ConfirmedCases']

y_fatal = train_dis['Fatalities']



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_confirm, test_size=0.3, random_state=0)



predict_labels = X.columns



#RF model

rf = RandomForestClassifier()



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")





# Print the name and gini importance of each feature

for feature in zip(predict_labels, rf.feature_importances_):

    print(feature)

plt.figure()



plt.title("Feature importances",fontsize=20)



plt.bar(predict_labels,rf.feature_importances_, align="center", color='dodgerblue')



plt.xticks(predict_labels)



plt.xticks(rotation=90)



plt.show()
## predict on test set



confirm_case = rf.predict(test_dis[['country_id','month','dates','distance_to_first_outbreak']])
#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_fatal, test_size=0.3, random_state=0)



predict_labels = X.columns



#RF model

rf = RandomForestClassifier()



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")

## predict on test set



fatal_case = rf.predict(test_dis[['country_id','month','dates','distance_to_first_outbreak']])



forecaseId = pd.DataFrame(test[['ForecastId']])

confirm_case = round(pd.DataFrame(confirm_case),0)

fatal_case = round(pd.DataFrame(fatal_case),0)





final_result_rf_id = pd.concat([forecaseId,confirm_case,fatal_case],axis=1)

final_result_rf_id.columns = ['ForecastId','ConfirmedCases','Fatalities']



final_result_rf_id.tail()

final_result_rf_id.to_csv('results_rf_dis_id.csv',index=False)
#run for each country

x_pred_lab = ['month','dates','distance_to_first_outbreak']



final_result_rf_dis_each = rf_each_country(seed=123, original_df=train, train_df=train_dis, test_df=test_dis, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_rf_dis_each.tail()
final_result_rf_dis_each.to_csv('submission.csv',index=False)
#run for each country

x_pred_lab = ['month','dates','distance_to_first_outbreak']



final_result_lin_dis_each = lin_each_country(seed=123, original_df=train, train_df=train_dis, test_df=test_dis, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_lin_dis_each.tail()
final_result_lin_dis_each.to_csv('final_result_lin_dis_each.csv',index=False)
#run for each country

x_pred_lab = ['month','dates','distance_to_first_outbreak']



final_result_br_dis_each = br_each_country(seed=123, original_df=train, train_df=train_dis, test_df=test_dis, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_br_dis_each.tail()
final_result_br_dis_each.to_csv('final_result_br_dis_each.csv',index=False)
train_demo1 = train_demo.copy()



# Add additional variables

#month

train_demo1['month'] = train_demo1['Date'].dt.month



#date

train_demo1['dates'] = train_demo1['Date'].dt.day





### do the same for test data

test_demo1 = test_demo.copy()



# Add additional variables



#month

test_demo1['month'] = test_demo1['Date'].dt.month



#date

test_demo1['dates'] = test_demo1['Date'].dt.day





test_demo1.tail()
### Interpolate missing values



train_demo1 = train_demo1.interpolate(method ='linear', limit_direction ='both')

test_demo1 = test_demo1.interpolate(method ='linear', limit_direction ='both')

#ecoding countries data

labels, values = pd.factorize(train_demo1['Country_Region'])



train_demo1['country_id'] = labels



labels, values = pd.factorize(test_demo1['Country_Region'])



test_demo1['country_id'] = labels



test_demo1.head()

### Train dataset



# Create x, where x the 'scores' column's values as floats

pop = train_demo1[['pop']].values.astype(float)

tests = train_demo1[['tests']].values.astype(float)

healthexp = train_demo1[['healthexp']].values.astype(float)



# Create a minimum and maximum processor object

min_max_scaler = preprocessing.MinMaxScaler()



# Create an object to transform the data to fit minmax processor

pop_scaled = min_max_scaler.fit_transform(pop)

test_scaled = min_max_scaler.fit_transform(tests)

healthexp_scaled = min_max_scaler.fit_transform(healthexp)



# Run the normalizer on the dataframe

train_demo1[['pop']] = pd.DataFrame(pop_scaled)

train_demo1[['tests']] = pd.DataFrame(test_scaled)

train_demo1[['healthexp']] = pd.DataFrame(healthexp_scaled)



train_demo1.head()



### Test dataset



# Create x, where x the 'scores' column's values as floats

pop = test_demo1[['pop']].values.astype(float)

tests = test_demo1[['tests']].values.astype(float)

healthexp = test_demo1[['healthexp']].values.astype(float)



# Create a minimum and maximum processor object

min_max_scaler = preprocessing.MinMaxScaler()



# Create an object to transform the data to fit minmax processor

pop_scaled = min_max_scaler.fit_transform(pop)

test_scaled = min_max_scaler.fit_transform(tests)

healthexp_scaled = min_max_scaler.fit_transform(healthexp)



# Run the normalizer on the dataframe

test_demo1[['pop']] = pd.DataFrame(pop_scaled)

test_demo1[['tests']] = pd.DataFrame(test_scaled)

test_demo1[['healthexp']] = pd.DataFrame(healthexp_scaled)



test_demo1.head()
random.seed(123)



X = train_demo1[['country_id','pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'month', 'dates']]



y_confirm = train_demo1['ConfirmedCases']

y_fatal = train_demo1['Fatalities']



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_confirm, test_size=0.3, random_state=0)



predict_labels = X.columns



#RF model

rf = RandomForestClassifier()



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)





print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")





# Print the name and gini importance of each feature

for feature in zip(predict_labels, rf.feature_importances_):

    print(feature)

    
plt.figure()



plt.title("Feature importances",fontsize=20)



plt.bar(predict_labels,rf.feature_importances_, align="center", color='dodgerblue')



plt.xticks(predict_labels)



plt.xticks(rotation=90)



plt.show()
## predict on test set



confirm_case = rf.predict(test_demo1[['country_id','pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'month', 'dates']])

random.seed(123)



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_fatal, test_size=0.3, random_state=0)



predict_labels = X.columns



#RF model

rf = RandomForestClassifier()



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")

    
## predict on test set



fatal_case = rf.predict(test_demo1[['country_id','pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'month', 'dates']])
forecaseId = pd.DataFrame(test[['ForecastId']])

confirm_case = pd.DataFrame(confirm_case)

fatal_case = pd.DataFrame(fatal_case)





final_result_rf_demo = pd.concat([forecaseId,confirm_case,fatal_case],axis=1)

final_result_rf_demo.columns = ['ForecastId','ConfirmedCases','Fatalities']



final_result_rf_demo.tail()
final_result_rf_demo.to_csv('result_rf_demo.csv',index=False)
#run for each country

x_pred_lab = ['pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'month', 'dates']



final_result_rf_demo_each = rf_each_country(seed=123, original_df=train, train_df=train_demo1, test_df=test_demo1, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_rf_demo_each.tail()
final_result_rf_demo_each.to_csv('final_result_rf_demo_each.csv',index=False)
#run for each country

x_pred_lab = ['pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'month', 'dates']



final_result_lin_demo_each = lin_each_country(seed=123, original_df=train, train_df=train_demo1, test_df=test_demo1, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_lin_demo_each.tail()
final_result_lin_demo_each.to_csv('final_result_lin_demo_each.csv',index=False)
#run for each country

x_pred_lab = ['pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'month', 'dates']



final_result_br_demo_each = br_each_country(seed=123, original_df=train, train_df=train_demo1, test_df=test_demo1, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_br_demo_each.tail()
final_result_br_demo_each.to_csv('final_result_br_demo_each.csv',index=False)
#construct the OLS model

X = train_demo1[[ 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'lung', 'femalelung', 'malelung','fertility']]

y = train_demo1['ConfirmedCases']



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.3, random_state=0)





# Note the difference in argument order

model = sm.OLS(y_train_rf, X_train_rf).fit()

predictions = model.predict(X_test_rf) # make the predictions by the model



model.summary()

#construct the OLS model

X = train_demo1[[ 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'lung', 'femalelung', 'malelung','fertility']]

y = train_demo1['Fatalities']



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.3, random_state=0)





# Note the difference in argument order

model = sm.OLS(y_train_rf, X_train_rf).fit()

predictions = model.predict(X_test_rf) # make the predictions by the model



model.summary()

weather_train = weather_train.drop(['Id'],axis=1)



weather_train = weather_train.rename(columns={'Id.1':'Id'})

weather_test1 = weather_test1.drop(['ForecastId'],axis=1)



weather_test1 = weather_test1.rename(columns={'ForecastId.1':'ForecastId'})

##combine dataframe



#train



train_all = pd.merge(train_first_result,train_demo1[['Id','Country_Region', 'Date', 

                                                     'pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'month', 'dates',

       'country_id']], on=['Id','Country_Region','Date','country_id','month', 'dates'], how='left')



train_all = pd.merge(train_all,weather_train[['Id','Country_Region', 'Date', 'Lat', 'Long',

       'day_from_jan_first', 'temp', 'min', 'max', 'stp', 'wdsp', 'prcp',

       'fog','month','dates']], on=['Id','Country_Region','Date','month','dates'], how='left')



train_all = pd.merge(train_all,train_dis[['Id','Country_Region', 'Date', 'month', 'dates', 'distance_to_first_outbreak',

       'country_id']], on=['Id','Country_Region','Date','country_id','month', 'dates'], how='left')



train_all = train_all.dropna(subset=['day_from_jan_first'])



train_all.tail()
#test



test_all = pd.merge(test_first_result,test_demo1[['ForecastId','Country_Region', 'Date', 

                                                     'pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'month', 'dates',

       'country_id']], on=['ForecastId','Country_Region','Date','country_id','month', 'dates'], how='left')



test_all = pd.merge(test_all,weather_test1[['ForecastId','Country_Region', 'Date',

       'Lat', 'Long',

       'day_from_jan_first', 'temp', 'min', 'max', 'stp', 'wdsp', 'prcp',

       'fog', 'month', 'dates', 'country_id']], on=['ForecastId','Country_Region',

                                                    'Date','country_id','month', 'dates'], how='left')



test_all = pd.merge(test_all,test_dis[['ForecastId','Country_Region', 'Date', 'month', 

                                          'dates', 'distance_to_first_outbreak',

       'country_id']], on=['ForecastId','Country_Region','Date','country_id','month', 'dates'], how='left')



test_all.tail()
random.seed(123)



X = train_all[['month', 'dates', 'days_since_first_outbreak',

       'country_id', 'pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'Lat',

       'Long', 'day_from_jan_first', 'temp', 'min', 'max', 'stp', 'wdsp',

       'prcp', 'fog', 'distance_to_first_outbreak']]

y_confirm = train_all['ConfirmedCases']

y_fatal = train_all['Fatalities']



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_confirm, test_size=0.3, random_state=0)



predict_labels = X.columns



#RF model

rf = RandomForestClassifier()



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")





# Print the name and gini importance of each feature

for feature in zip(predict_labels, rf.feature_importances_):

    print(feature)
plt.figure()



plt.title("Feature importances",fontsize=20)



plt.bar(predict_labels,rf.feature_importances_, align="center", color='dodgerblue')



plt.xticks(predict_labels)



plt.xticks(rotation=90)



plt.show()
## predict on test set



confirm_case = rf.predict(test_all[['month', 'dates', 'days_since_first_outbreak',

       'country_id', 'pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'Lat',

       'Long', 'day_from_jan_first', 'temp', 'min', 'max', 'stp', 'wdsp',

       'prcp', 'fog', 'distance_to_first_outbreak']])

random.seed(123)



#Now we find the best parameters to fit in the Random Forest model, we will use it to measure the feature important in the data

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_fatal, test_size=0.3, random_state=0)



predict_labels = X.columns



#RF model

rf = RandomForestClassifier()



# Train the classifier

rf.fit(X_train_rf, y_train_rf)



y_pred_rf = rf.predict(X_test_rf)



print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_rf, y_pred_rf),2))  

print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_rf, y_pred_rf),2))  

print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_rf, y_pred_rf)),2), "\n")

    

fatal_case = rf.predict(test_all[['month', 'dates', 'days_since_first_outbreak',

       'country_id', 'pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'Lat',

       'Long', 'day_from_jan_first', 'temp', 'min', 'max', 'stp', 'wdsp',

       'prcp', 'fog', 'distance_to_first_outbreak']])

forecaseId = pd.DataFrame(test[['ForecastId']])

confirm_case = pd.DataFrame(confirm_case)

fatal_case = pd.DataFrame(fatal_case)





final_result_all = pd.concat([forecaseId,confirm_case,fatal_case],axis=1)

final_result_all.columns = ['ForecastId','ConfirmedCases','Fatalities']



final_result_all.tail()
final_result_all.to_csv('result_all_rf.csv',index=False)
#run for each country

x_pred_lab = ['month', 'dates', 'days_since_first_outbreak',

       'pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'Lat',

       'Long', 'day_from_jan_first', 'temp', 'min', 'max', 'stp', 'wdsp',

       'prcp', 'fog', 'distance_to_first_outbreak']



final_result_rf_all_each = rf_each_country(seed=123, original_df=train, train_df=train_all, test_df=test_all, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_rf_all_each.tail()

final_result_rf_all_each.to_csv('final_result_rf_all_each.csv',index=False)
#run for each country

x_pred_lab = ['month', 'dates', 'days_since_first_outbreak',

       'pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'Lat',

       'Long', 'day_from_jan_first', 'temp', 'min', 'max', 'stp', 'wdsp',

       'prcp', 'fog', 'distance_to_first_outbreak']



final_result_lin_all_each = lin_each_country(seed=123, original_df=train, train_df=train_all, test_df=test_all, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_lin_all_each.tail()
final_result_lin_all_each.to_csv('final_result_lin_all_each.csv',index=False)
#run for each country

x_pred_lab = ['month', 'dates', 'days_since_first_outbreak',

       'pop', 'tests', 'testpop', 'density', 'medianage',

       'urbanpop', 'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54',

       'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung',

       'healthexp', 'healthperpop', 'fertility', 'Lat',

       'Long', 'day_from_jan_first', 'temp', 'min', 'max', 'stp', 'wdsp',

       'prcp', 'fog', 'distance_to_first_outbreak']



final_result_br_all_each = br_each_country(seed=123, original_df=train, train_df=train_all, test_df=test_all, 

                        confirm ='ConfirmedCases', fatal='Fatalities', xlabels = x_pred_lab, test_size = 0.3)

final_result_br_all_each.tail()
final_result_br_all_each.to_csv('final_result_br_all_each.csv',index=False)