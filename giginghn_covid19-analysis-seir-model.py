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

from matplotlib import dates

import plotly.graph_objects as go

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline



### Load in the data from Kaggle Week 2



train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv",parse_dates=['Date'])

                    

train.tail()
train.info()
# Test dataset from Kaggle



test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv",parse_dates=['Date'])

                    

test.tail()

submit = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

                     

submit.head()
#read the complete data set



complete_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])



complete_data = complete_data.rename(columns = {'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})



complete_data['Active'] = complete_data['Confirmed'] - complete_data['Deaths'] - complete_data['Recovered']



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

weather_data = pd.read_csv("../input/weatherweek4/training_data_with_weather_info_week_4.csv", parse_dates=['Date'])

weather_test = pd.read_csv("../input/weatherweek4/testing_data_with_weather_info_week_4.csv", parse_dates=['Date'])

weather_data.info()


weather_data.head()



weather_data = weather_data.replace(9999.9,np.NaN)



weather_data = weather_data.replace(999.9,np.NaN)



weather_data = weather_data.replace(99.99,np.NaN)



weather_data.head()

#interpolate missing value with linear method



weather_data['stp'] = weather_data['stp'].interpolate(method ='linear', limit_direction ='both') 

weather_data['wdsp'] = weather_data['wdsp'].interpolate(method ='linear', limit_direction ='both') 

weather_data['prcp'] = weather_data['prcp'].interpolate(method ='linear', limit_direction ='both') 



weather_data.sort_values(['Date']).tail()

## From @winterpierre source



weather_addition = pd.read_csv("../input/covid19-global-weather-data/temperature_dataframe.csv", parse_dates=['date'])



#rename column

weather_addition.columns = ['Unnamed: 0', 'Id', 'Province_State', 'Country_Region', 'lat', 'long', 'Date',

       'ConfirmedCases', 'Fatalities', 'capital', 'humidity', 'sunHour', 'tempC',

       'windspeedKmph']



#fix the name US for consistency

weather_addition = weather_addition.replace('USA','US')



weather_addition.head()



lockdown = pd.read_csv('../input/covid19-lockdown-dates-by-country/countryLockdowndates.csv', parse_dates=['Date'])



lockdown = lockdown.replace(pd.to_datetime('2020-11-03'),pd.to_datetime('2020-03-11'))





lockdown.head()



high_risk_eu = ['Germany','Spain','Italy','France']



lockdown1 = lockdown.loc[lockdown['Country/Region'].isin(high_risk_eu)==True]



lockdown1.sort_values(['Date'])

high_risk_us = ['US']



lockdown2 = lockdown.loc[lockdown['Country/Region'].isin(high_risk_us)==True]



lockdown2.sort_values(['Date'])

high_risk_us = ['Iran','Turkey','Russia','India','Korea, South']



lockdown3 = lockdown.loc[lockdown['Country/Region'].isin(high_risk_us)==True]



lockdown3

test_conduct = pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_Conducted_31Mar2020.csv')



test_conduct.head()
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



fig.update_layout(title="COVID19 Confirmed Cases Globally")



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

spain_sum = complete_data.loc[complete_data['Country_Region']=="Spain"].groupby('Date')['Date', 'Confirmed','Deaths','Active','Recovered'].sum().reset_index()



spain_sum = pd.melt(spain_sum, id_vars=['Date'], value_vars=['Confirmed','Deaths','Active','Recovered'])



spain_sum = spain_sum.rename(columns={"value": "total", "variable": "Cases"})



sns.lineplot(data=spain_sum, x="Date", y="total", hue="Cases")



plt.ylabel("Total cases in Spain")



plt.title("Total numbers of Cases in Spain",fontsize=20)

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



plt.title("Fatality Rate in Asia countries - excluding China and Iran",fontsize=20)


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

prcp_covid = weather_data.groupby(['Date', 'Country_Region'])['prcp'].max().reset_index()

prcp_covid['Date'] = pd.to_datetime(prcp_covid['Date'])



#merge with the confirmed cases for size changing

prcp_covid = pd.merge(prcp_covid, map_covid, on=['Date','Country_Region'],how='left')



prcp_covid['Date'] = prcp_covid['Date'].dt.strftime('%m/%d/%Y')



fig = px.scatter_geo(prcp_covid, locations="Country_Region", locationmode='country names', 

                     color="prcp", size='size', hover_name="Country_Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Precipitation according to the number of Confirmed Cases Around the Globe', 

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

temp_covid1 = pd.merge(temp_covid1, prcp_covid[['Date','Country_Region','prcp']], 

                             on=['Date','Country_Region'],how='left')

temp_covid1 = pd.merge(temp_covid1, humid_covid[['Date','Country_Region','humidity']], 

                             on=['Date','Country_Region'],how='left')

temp_covid1 = pd.merge(temp_covid1, sun_covid[['Date','Country_Region','sunHour']], 

                             on=['Date','Country_Region'],how='left')



temp_covid1 = temp_covid1.dropna()



temp_covid1.tail()
temp_covid2 = temp_covid1[['temp','wdsp','prcp','humidity','sunHour']]



corr = temp_covid2.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



sns.set(style="white")



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



#construct the Multilinear regression model

X =  temp_covid1[['temp','wdsp','humidity']]

y = temp_covid1['ConfirmedCases']



# Note the difference in argument order

model = sm.OLS(y, X).fit()



#model summary

model.summary()

china_temp = temp_covid1.loc[temp_covid1['Country_Region']=='China']



#construct the OLS model

X =  china_temp[['temp', 'wdsp', 'humidity']]

y = china_temp['ConfirmedCases']



# Note the difference in argument order

model = sm.OLS(y, X).fit()



#predictions = model.predict(X) # make the predictions by the model



model.summary()

italy_temp = temp_covid1.loc[temp_covid1['Country_Region']=='Italy']



#construct the OLS model

X =  italy_temp[['temp', 'wdsp', 'humidity']]

y = italy_temp['ConfirmedCases']



# Note the difference in argument order

model = sm.OLS(y, X).fit()



#predictions = model.predict(X) # make the predictions by the model



model.summary()
spain_temp = temp_covid1.loc[temp_covid1['Country_Region']=='Spain']



#construct the OLS model

X =  spain_temp[['temp', 'wdsp', 'humidity']]

y = spain_temp['ConfirmedCases']



# Note the difference in argument order

model = sm.OLS(y, X).fit()



#predictions = model.predict(X) # make the predictions by the model



model.summary()


us_temp = temp_covid1.loc[temp_covid1['Country_Region']=='US']



#construct the OLS model

X =  us_temp[['temp','wdsp', 'humidity']]

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



sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

sns.set(rc={'figure.figsize':(15, 7)})



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
Image("../input/images/log-curve.png")
china_sum = complete_data.loc[complete_data['Country_Region']=="China"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



china_sum = pd.melt(china_sum, id_vars=['Date'], value_vars=['Confirmed'])



china_sum = china_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=china_sum, x="Date", y="total", color="dodgerblue")



ax.axvline(pd.to_datetime('2020-01-23'), color="red", linestyle="--")



ax.axvline(pd.to_datetime('2020-02-12'), color="gray", linestyle="--")



ax.annotate("Date first quarantine", xy=(pd.to_datetime('2020-01-24'), 50000))



ax.annotate("Inflection Point", xy=(pd.to_datetime('2020-02-13'), 50000))



plt.ylabel("Total cases in China")



plt.title("Total numbers of Cases in China",fontsize=20)

korea_sum = complete_data.loc[complete_data['Country_Region']=="South Korea"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



korea_sum = pd.melt(korea_sum, id_vars=['Date'], value_vars=['Confirmed'])



korea_sum = korea_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=korea_sum, x="Date", y="total", color="violet")



ax.axvline(pd.to_datetime('2020-02-23'), color="black", linestyle="--")



ax.axvline(pd.to_datetime('2020-03-01'), color="gray", linestyle="--")



ax.axvline(pd.to_datetime('2020-02-01'), color="red", linestyle="--")



ax.annotate("First roll out \n widespread testing", xy=(pd.to_datetime('2020-02-02'), 5000))



ax.annotate("Date First \nQuarantine", xy=(pd.to_datetime('2020-02-15'), 7000))



ax.annotate("Inflection Point", xy=(pd.to_datetime('2020-03-03'), 4000))



plt.ylabel("Total confirmed cases in South Korea")



plt.title("Total numbers of Cases in South Korea",fontsize=20)

italy_sum = complete_data.loc[complete_data['Country_Region']=="Italy"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



italy_sum = pd.melt(italy_sum, id_vars=['Date'], value_vars=['Confirmed'])



italy_sum = italy_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=italy_sum, x="Date", y="total", color="palevioletred")



ax.axvline(pd.to_datetime('2020-03-11'), color="red", linestyle="--")



ax.annotate("Date First Quarantine", xy=(pd.to_datetime('2020-03-12'), 125000))



plt.ylabel("Total confirmed cases in Italy")



plt.title("Total numbers of Cases in Italy",fontsize=20)

spain_sum = complete_data.loc[complete_data['Country_Region']=="Spain"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



spain_sum = pd.melt(spain_sum, id_vars=['Date'], value_vars=['Confirmed'])



spain_sum = spain_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=spain_sum, x="Date", y="total", color="green")



ax.axvline(pd.to_datetime('2020-03-14'), color="red", linestyle="--")



ax.annotate("Date First Quarantine", xy=(pd.to_datetime('2020-03-15'), 125000))



plt.ylabel("Total confirmed cases in Spain")



plt.title("Total numbers of Cases in Spain",fontsize=20)

france_sum = complete_data.loc[complete_data['Country_Region']=="France"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



france_sum = pd.melt(france_sum, id_vars=['Date'], value_vars=['Confirmed'])



france_sum = france_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=france_sum, x="Date", y="total", color="salmon")



ax.axvline(pd.to_datetime('2020-03-16'), color="red", linestyle="--")



ax.annotate("Date First Quarantine", xy=(pd.to_datetime('2020-03-17'), 110000))



plt.ylabel("Total confirmed cases in France")



plt.title("Total numbers of Cases in France",fontsize=20)

germany_sum = complete_data.loc[complete_data['Country_Region']=="Germany"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



germany_sum = pd.melt(germany_sum, id_vars=['Date'], value_vars=['Confirmed'])



germany_sum = france_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=germany_sum, x="Date", y="total", color="orange")



ax.axvline(pd.to_datetime('2020-03-20'), color="red", linestyle="--")



ax.annotate("Date First Quarantine", xy=(pd.to_datetime('2020-03-21'), 125000))



plt.ylabel("Total confirmed cases in Germany")



plt.title("Total numbers of Cases in Germany",fontsize=20)

iran_sum = complete_data.loc[complete_data['Country_Region']=="Iran"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



iran_sum = pd.melt(iran_sum, id_vars=['Date'], value_vars=['Confirmed'])



iran_sum = iran_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=iran_sum, x="Date", y="total", color="cornflowerblue")



ax.axvline(pd.to_datetime('2020-03-15'), color="red", linestyle="--")



ax.annotate("Date First Quarantine", xy=(pd.to_datetime('2020-03-16'), 60000))



plt.ylabel("Total confirmed cases in Iran")



plt.title("Total numbers of Cases in Iran",fontsize=20)

us_sum = complete_data.loc[complete_data['Country_Region']=="US"].groupby('Date')['Date', 'Confirmed'].sum().reset_index()



us_sum = pd.melt(us_sum, id_vars=['Date'], value_vars=['Confirmed'])



us_sum = us_sum.rename(columns={"value": "total", "variable": "Cases"})



ax=sns.lineplot(data=us_sum, x="Date", y="total", color="orchid")



ax.axvline(pd.to_datetime('2020-03-22'), color="red", linestyle="--")



ax.annotate("Date First Quarantine \nin New York", xy=(pd.to_datetime('2020-03-23'), 600000))



plt.ylabel("Total confirmed cases in the U.S.")



plt.title("Total numbers of Cases in the U.S.",fontsize=20)

Image("../input/images/SEIR-math.png")

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
def fit_model_new(data, area_name, initial_guess=[2.2, 0.02, 2, 50], 

              bounds=((1, 20), (0, 0.15), (1, 3), (1, 100)), make_plot=True, decay_mode = None):

    

    if area_name in ['France']:# France last data looks weird, remove it

        train = data.query('ConfirmedCases > 0').copy()[:-1]

    else:

        train = data.query('ConfirmedCases > 0').copy()

    

    ####### Split Train & Valid #######

    valid_data = train[-7:].copy()

    train_data = train[:-7].copy()

    

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

        

    if area_name == 'Global':

        population = 7744240900

        

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

    test_end = datetime.strptime('2020-05-14','%Y-%m-%d')

    test_start = datetime.strptime('2020-04-02','%Y-%m-%d')

    test_period = (test_end - test_start).days

    train_max = train_data.Date.max()

    train_all_max = train.Date.max()

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

    

    ####### Plot the fit result of train data and forecast after 300 days ####### 

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

            

        ####### Forecast 300 days after by using the best paramater of train data #######

        if len(res.x)>2:

            msle, sol = eval_model_decay(res.x, train_data, population, True, 300)

        else:

            msle, sol = eval_model_const(res.x, train_data, population, True, 300)

        

        sus, exp, inf, rec = sol.y

        

        y_pred = pd.DataFrame({

            'ConfirmedCases': cumsum_signal(np.diff((inf + rec) * population, prepend=n_infected).cumsum()),

            'Fatalities': cumsum_signal(np.clip(rec, 0, np.inf) * population * res.x[1])

        })

        

        ####### Plot 300 days after of each country #######

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

                            name='Confirmed case until '+ str(train_all_max.date()),line = dict(color='green', width=4)))

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
country = 'US'

country_pd_train = train[train['Country_Region']==country]

country_pd_train2 = country_pd_train.groupby(['Date']).sum().reset_index()

country_pd_train2['Date'] = pd.to_datetime(country_pd_train2['Date'], format='%Y-%m-%d')

a,b = fit_model_new(country_pd_train2,country,make_plot=True)

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