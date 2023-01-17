from __future__ import print_function

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets



import numpy as np

import pandas as pd

import datetime as dt

from datetime import date



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import folium #to create maps





from sklearn.linear_model import LinearRegression

from sklearn import metrics



#for nonlinear regression

from scipy.optimize import curve_fit



#%%writefile covid19_ascending_model_functions.py



# %load covid19_ascending_model_functions.py

"""

Created on Thu Mar 26 14:18:40 2020



@author: Raili

"""



#Date is transformed into a column. This makes plotting the time series data easeier. 



def add_date_column(df_raw):

    df_date_column = df_raw.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_name='Cases', var_name='Date')

    df_date_column = df_date_column.set_index(['Country/Region', 'Province/State', 'Date'])

    return df_date_column



#function that takes raw data for confirmed, deaths and recovered. Returns dataframe with 

#current accumulated data for confirmed, deaths and recovered.  Basically, it removes

#dates and returns totals for each country at the most recently reported data (last column)



#example function call

#accum_data_df = accum_data(raw_data_confirmed, raw_data_recovered, raw_data_deaths)

#raw_data_confirmed (and others) have following header:

#Province/State, Country/Region, Lat, Long, 1/22/20,...3/18/20



#Dataframe that is returned has following header:

#Province/State, Country, Lat, Long, Total Confirmed, Total Deaths, Total Recovered

def accumulated_data(confirmed, deaths, recovered):

        

    accum_data_df = confirmed[['Province/State', 'Country/Region', 'Lat', 'Long']]

    

    accum_data_df['Total Confirmed'] = confirmed.iloc[ :, -1]

    accum_data_df['Total Deaths'] = deaths.iloc[ :, -1]

    accum_data_df['Total Recovered'] = recovered.iloc[ :, -1]



    return accum_data_df





### Get DailyData from Cumulative sum

def daily_data(country_time_df,old_name,new_name):

    accum_country_daily=country_time_df.groupby(level=0).diff().fillna(0)

    accum_country_daily=accum_country_daily.rename(columns={old_name:new_name})

    return accum_country_daily



#Function that takes data with date column and returns a dataframe with countries, dates, confirmed, deaths and recovered.

#type_data is either Country/Region or Province/State. 

#Example function call: 

#timeseries_country_data(confirmed_date_column,'Cases','Total Confirmed Cases', 'Country')

#confirmed_date_column has following header:

#Country/Region, Province/State, Date, Lat, Long, Cases



#Dateframe that is returned has following header:

#Country/Region, Date, ......, newname



def timeseries_data(_time_df,old_name,new_name, type_data):

        

    timeseries_df=_time_df.groupby([type_data,'Date'])['Cases'].sum().reset_index()

    timeseries_df=timeseries_df.set_index([type_data,'Date'])

    timeseries_df.index=timeseries_df.index.set_levels([timeseries_df.index.levels[0], pd.to_datetime(timeseries_df.index.levels[1])])

    timeseries_df=timeseries_df.sort_values([type_data,'Date'],ascending=True)

    timeseries_df=timeseries_df.rename(columns={old_name:new_name})

    return timeseries_df



#Function that takes data with date column and returns a dataframe with states, dates, confirmed, deaths and recovered. 

#Example function call: 

#timeseries_state_data(confirmed_date_column, country, 'Cases','Total Confirmed Cases')

#country is the country that you want state data

#confirmed_date_column has following header:

#Country/Region, Province/State, Date, Lat, Long, Cases



#Dateframe that is returned has following header:

#State, Date, ......, newname



def timeseries_state_data(state_time_df,old_name,new_name):

    state_time_df=state_time_df.groupby(['Province/State','Date'])['Cases'].sum().reset_index()

    state_time_df=state_time_df.set_index(['Province/State','Date'])

    state_time_df.index=state_time_df.index.set_levels([state_time_df.index.levels[0], pd.to_datetime(state_time_df.index.levels[1])])

    state_time_df=state_time_df.sort_values(['Province/State','Date'],ascending=True)

    state_time_df=state_time_df.rename(columns={old_name:new_name})

    

    return state_time_df



#function to clean the data

def clean_data(df_to_clean):

    

    # replace mainland china with China

    df_to_clean['Country/Region'] = df_to_clean['Country/Region'].replace('Mainland China', 'China')

    

    return df_to_clean

    

    

#-----------------------------------------------------------------------------------------------------------------

#function calculates likely positive cases based on current death rate, mortality rate, and doubling time

#consolidated_df: dataframe that includes columns named 'Daily New Deaths'



def estimate_current_cases(consolidated_df, days_to_symptoms, days_after_symptoms, mortality_rate, doubling_time, place):

    

    death_time = days_to_symptoms + days_after_symptoms



    #calculate estimated people who have virus at the time of the death. For instance if death_time is 

    #20 days this calculation shows the possible number of positive cases 20 days previously.

    #Since these are calculated for the number of days before death shift data in dataframe back by death_time

    #days.

    estimated_past_cases = consolidated_df.copy()

    estimated_past_cases = estimated_past_cases.loc[:,['Daily New Deaths']]

    estimated_past_cases = np.round(estimated_past_cases['Daily New Deaths']/mortality_rate,3)



    #calculate possible number of cases at current date with the doubling rate. 

    #estimated_current_cases = consolidated_df.copy()

    #estimated_current_cases = estimated_current_cases.loc[:,['Daily New Deaths']]

    estimated_current_cases = np.round(estimated_past_cases*2**(death_time/doubling_time))



    #Estimated past cases were calculated for the number of days death_time previously. They need to be shifted back

    #by this number of days. 

    estimated_past_cases = estimated_past_cases.groupby([place]).shift(-death_time)



    #Combine estimated past cases with estimated current cases

    data_to_combine = [estimated_past_cases, estimated_current_cases]

    headers = ['Estimated Past Cases', 'Estimated Current Cases']

    estimated_cases = pd.concat(data_to_combine, axis=1, keys=headers)

    

    return estimated_cases

url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'



raw_data_confirmed = pd.read_csv(url_confirmed)

raw_data_deaths = pd.read_csv(url_deaths)

raw_data_recovered = pd.read_csv(url_recovered)

url_confirmed_state = 'https://www.soothsawyer.com/wp-content/uploads/2020/03/time_series_19-covid-Confirmed.csv'

url_deaths_state = 'https://www.soothsawyer.com/wp-content/uploads/2020/03/time_series_19-covid-Deaths.csv'

url_recovered_state = 'https://www.soothsawyer.com/wp-content/uploads/2020/03/time_series_19-covid-Recovered.csv'



raw_data_cases_state = pd.read_csv(url_confirmed_state)

raw_data_deaths_state = pd.read_csv(url_deaths_state)

raw_data_recovered_state = pd.read_csv(url_recovered_state)



raw_data_cases_state
#Combine data into a table with accumulated data. This is basically the last column of the 

#raw data set for confirmed cases, recovered cases and deaths. 

accum_data_df = accumulated_data(raw_data_confirmed, raw_data_deaths, raw_data_recovered)

accum_data_df.head()
country_table = accum_data_df.groupby('Country/Region')['Total Confirmed','Total Deaths', 'Total Recovered'].sum()

country_table_sorted = country_table.sort_values(by='Total Confirmed', ascending=False)

country_table_sorted.style.background_gradient()
#Remove cases from dataset that are zero for 'Total Confirmed' for map

map_data = accum_data_df[accum_data_df['Total Confirmed']>0]



#map of Total Confirmed Cases

covid_map = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=6, zoom_start=2)



for i in range(0, len(map_data)):

    folium.CircleMarker(

        location=[map_data.iloc[i]['Lat'], map_data.iloc[i]['Long']],

        radius=int(map_data.iloc[i]['Total Confirmed'])/5000,

        color='blue',

        tooltip =   '<li><bold>Country : '+str(map_data.iloc[i]['Country/Region'])+

                    '<li><bold>Province : '+str(map_data.iloc[i]['Province/State'])+

                    '<li><bold>Confirmed : '+str(map_data.iloc[i]['Total Confirmed'])+

                    '<li><bold>Deaths : '+str(map_data.iloc[i]['Total Deaths'])

         ).add_to(covid_map)

    

covid_map

#Remove cases from dataset that are zero for 'Total Deaths' for map

map_data_deaths = map_data[map_data['Total Deaths']>0]



#map of Total Confirmed Deaths

covid_map_deaths = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=6, zoom_start=2)



for i in range(0, len(map_data_deaths)):

    folium.CircleMarker(

        location=[map_data_deaths.iloc[i]['Lat'], map_data_deaths.iloc[i]['Long']],

        radius=int(map_data_deaths.iloc[i]['Total Deaths'])/1000,

        color='black',

        tooltip =   '<li><bold>Deaths : '+str(map_data_deaths.iloc[i]['Total Deaths'])).add_to(covid_map_deaths)      

        

covid_map_deaths
#Remove cases from dataset that are zero for 'Total Deaths' for map

map_data_deaths = map_data[map_data['Total Deaths']>0]

#Convert dates from raw data from column headers to a column for easier plotting. 

#Do this for the confirmed, deaths and recovered data sets. 

confirmed_date_column=add_date_column(raw_data_confirmed)

deaths_date_column=add_date_column(raw_data_deaths)

#recovered_date_column=add_date_column(raw_data_recovered)



confirmed_date_column.head()
#Consolidate dates associated with each country for confirmed, deaths and recovered

confirmed_country_df=timeseries_data(confirmed_date_column,'Cases','Total Confirmed Cases', 'Country/Region')

deaths_country_df=timeseries_data(deaths_date_column,'Cases','Total Deaths','Country/Region')

#recoveries_country_df=timeseries_data(recovered_date_column,'Cases','Total Recoveries','Country/Region')



confirmed_country_df.head()
#Calculate daily data for confirmed, deaths and recoveries. This is the difference of data between

#each successive day. 

new_daily_cases_country=daily_data(confirmed_country_df,'Total Confirmed Cases','Daily New Cases')

new_daily_deaths_country=daily_data(deaths_country_df,'Total Deaths','Daily New Deaths')

#new_daily_recoveries_country=daily_data(recoveries_country_df,'Total Recoveries','Daily New Recoveries')



new_daily_cases_country.head()
#Combine the date, country and daily values for confirmed, deaths and recovered into one dataframe. 

country_consolidated_df=pd.merge(confirmed_country_df,deaths_country_df,how='left',left_index=True,right_index=True)

#country_consolidated_df=pd.merge(country_consolidated_df,recoveries_country_df,how='left',left_index=True,right_index=True)

country_consolidated_df=pd.merge(country_consolidated_df,new_daily_cases_country,how='left',left_index=True,right_index=True)

country_consolidated_df=pd.merge(country_consolidated_df,new_daily_deaths_country,how='left',left_index=True,right_index=True)

#country_consolidated_df=pd.merge(country_consolidated_df,new_daily_recoveries_country,how='left',left_index=True,right_index=True)



#Calculate current active cases and add as new column in dataframe

#country_consolidated_df['Active Cases']=country_consolidated_df['Total Confirmed Cases']-country_consolidated_df['Total Deaths']



#Calculate percent recoveries

#country_consolidated_df['Share of Recoveries - Closed Cases']=np.round(country_consolidated_df['Total Recoveries']/(country_consolidated_df['Total Recoveries']+country_consolidated_df['Total Deaths']),2)



#Calculate mortality rate by using total deaths divided total confirmed cases. Note that this mortality rate

#is likely higher than the actual mortality rate because there are cases that aren't counted if the person wasn't

#tested. Also, this is only one way to calculate the mortality rate. There are also people among the total confirmed

#cases who have not yet died or recovered. This number should be considered a best guessed estimate based on 

#available data. 

country_consolidated_df['Death to Cases Ratio']=np.round(country_consolidated_df['Total Deaths']/country_consolidated_df['Total Confirmed Cases'],3)



country_consolidated_df.head()
#Estimated parameters

#The incubation period (time from exposure to the development of symptoms) of the virus is estimated to be between 2 and 14 days based on the following sources:

#

#    The World Health Organization (WHO) reported an incubation period for COVID-19 between 2 and 10 days. [1]

#    China’s National Health Commission (NHC) had initially estimated an incubation period from 10 to 14 days [2].

#    The United States' CDC estimates the incubation period for COVID-19 to be between 2 and 14 days [3].

#    DXY.cn, a leading Chinese online community for physicians and health care professionals, is reporting an incubation period of "3 to 7 days, up to 14 days".

#

#The estimated range will be most likely narrowed down as more data becomes available.

days_to_symptoms = 6 #number of days until someone who was exposed shows symptoms



#Average person dies within 14 days after onset of symptoms

#https://onlinelibrary.wiley.com/doi/pdf/10.1002/jmv.25689

days_after_symptoms = 14 #number of days after someone who shows symptoms dies



#time to death is time from exposure that symptoms first appeared, and then time to death 

#after symptoms

death_time = days_to_symptoms + days_after_symptoms



#Current worldwide average is around 1%, however it was higher in China (3.4%) and other countries. 

#Since testing is still not widespread in many places it's hard to know what the true mortality rate is 

#at this point. 

mortality_rate = 0.01 #percent



#There are also various estimates for the doubling rate. The data from the US and Italy indicates that the 

#doubling time is approximately 5 days. 

doubling_time = 3 #days for cases to double



estimated_cases = estimate_current_cases(country_consolidated_df, days_to_symptoms, days_after_symptoms, mortality_rate, doubling_time, 'Country/Region')

#Add estimated cases to country_consolidated_df

country_consolidated_df = pd.merge(country_consolidated_df, estimated_cases, how='left', left_index=True, right_index=True)



#pd.set_option('display.max_rows', 12000)

#country_consolidated_df
#Place is an array containing a list of countries or a list of states

#df is a dataframe containing the following columns: Total Confirmed Cases, Total Deaths, Death to Case Ratio, 

#estimated_current_cases, estimated_past_cases



def plot_country_data(Country):

    fig = make_subplots(rows=6, cols=1,shared_xaxes=False, 

                    subplot_titles=('Daily Confirmed Cases', 'Accumulative Confirmed Cases', 'Daily Deaths', 'Accumulative Deaths', 'Estimated Mortality Rate', 'Estimated Current Cases'))

    fig.add_trace(go.Bar(x=country_consolidated_df.loc[Country].index,y=country_consolidated_df.loc[Country, 'Daily New Cases'],

                    name='Daily Confirmed Cases'),

                    row=1, col=1)

    fig.add_trace(go.Scatter(x=country_consolidated_df.loc[Country].index,y=country_consolidated_df.loc[Country, 'Total Confirmed Cases'],

                    name='Accumulative Confirmed Cases',

                    line=dict(color='firebrick', width=3, dash='dot')),

                    row=2, col=1)

    fig.add_trace(go.Bar(x=country_consolidated_df.loc[Country].index,y=country_consolidated_df.loc[Country,'Daily New Deaths'],

                    name='Daily Deaths'),

                    row=3, col=1)

    fig.add_trace(go.Scatter(x=country_consolidated_df.loc[Country].index,y=country_consolidated_df.loc[Country,'Total Deaths'],

                    name='Acculative Deaths',

                    line=dict(color='firebrick', width=3, dash='dot')),

                    row=4, col=1)

    fig.add_trace(go.Scatter(x=country_consolidated_df.loc[Country].index,y=country_consolidated_df.loc[Country,'Death to Cases Ratio'],

                    mode='lines+markers',

                    name='Estimated Mortality Rate',

                    line=dict(color='LightSkyBlue',width=2)),

                    row=5,col=1)

    fig.add_trace(go.Bar(x=country_consolidated_df.loc[Country].index,y=country_consolidated_df.loc[Country,'Estimated Current Cases'],

                    name='Estimated Current Cases'),

                    row=6,col=1)

    fig.append_trace(go.Bar(x=country_consolidated_df.loc[Country].index,y=country_consolidated_df.loc[Country,'Estimated Past Cases'],

                    name='Estimated Current Cases'),

                    row=6,col=1)

    

    fig.update_layout(height=1200, width=800,showlegend=False)

    return fig
#Plot country data.  



CountriesList=['US','Italy','Spain','Germany','Iran','China','France','Switzerland','United Kingdom','Netherlands','Austria','Belgium','Norway','Sweden','Korea, South']

interact(plot_country_data, Country=widgets.Dropdown(options=CountriesList))
mean_death_to_cases_ratio = country_consolidated_df.groupby('Country/Region', as_index=True)['Death to Cases Ratio'].agg([np.mean])

mean_death_to_cases_ratio = mean_death_to_cases_ratio.dropna()

mean_death_to_cases_ratio = mean_death_to_cases_ratio[mean_death_to_cases_ratio['mean']>0]

mean_death_to_cases_ratio = mean_death_to_cases_ratio[mean_death_to_cases_ratio['mean']<0.08]
#Table showing mortality rate for each state or province

temp = mean_death_to_cases_ratio

temp_sorted = temp.sort_values(by=['mean'], ascending=False)

temp_sorted.style.background_gradient()
histogram_x = mean_death_to_cases_ratio['mean']

plt.hist(histogram_x, bins='auto')

plt.xlabel('Mortality Rate')

plt.ylabel('Frequency')

plt.show()
mean_death_to_cases_ratio.mean()
#Convert dates from raw data from column headers to a column for easier plotting. 

#Do this for the confirmed, deaths and recovered data sets. 

confirmed_date_column_state = add_date_column(raw_data_cases_state)

deaths_date_column_state = add_date_column(raw_data_deaths_state)

recovered_date_column_state = add_date_column(raw_data_recovered_state)



confirmed_date_column_state.head()
#consolidate state data for timeseries plots



confirmed_state_df=timeseries_data(confirmed_date_column_state,'Cases','Total Confirmed Cases','Province/State')

deaths_state_df=timeseries_data(deaths_date_column_state,'Cases','Total Deaths','Province/State')

recoveries_state_df=timeseries_data(recovered_date_column_state,'Cases','Total Recoveries','Province/State')



confirmed_state_df.head()
#get new daily data from cumulative sum

current_cases_state=daily_data(confirmed_state_df,'Total Confirmed Cases','Daily New Cases')

current_deaths_state=daily_data(deaths_state_df,'Total Deaths','Daily New Deaths')

current_recoveries_state=daily_data(recoveries_state_df,'Total Recoveries','Daily New Recoveries')



confirmed_state_df
#combine all data sets

state_consolidated_df=pd.merge(confirmed_state_df,deaths_state_df,how='left',left_index=True,right_index=True)

state_consolidated_df=pd.merge(state_consolidated_df,recoveries_state_df,how='left',left_index=True,right_index=True)

state_consolidated_df=pd.merge(state_consolidated_df,current_cases_state,how='left',left_index=True,right_index=True)

state_consolidated_df=pd.merge(state_consolidated_df,current_deaths_state,how='left',left_index=True,right_index=True)

state_consolidated_df=pd.merge(state_consolidated_df,current_recoveries_state,how='left',left_index=True,right_index=True)

state_consolidated_df['Active Cases']=state_consolidated_df['Total Confirmed Cases']-state_consolidated_df['Total Deaths']-state_consolidated_df['Total Recoveries']

state_consolidated_df['Share of Recoveries - Closed Cases']=np.round(state_consolidated_df['Total Recoveries']/(state_consolidated_df['Total Recoveries']+state_consolidated_df['Total Deaths']),2)

state_consolidated_df['Death to Cases Ratio']=np.round(state_consolidated_df['Total Deaths']/state_consolidated_df['Total Confirmed Cases'],3)



estimated_cases_state = estimate_current_cases(state_consolidated_df, days_to_symptoms, days_after_symptoms, mortality_rate, doubling_time,'Province/State')

#Add estimated cases to country_consolidated_df

state_consolidated_df = pd.merge(state_consolidated_df, estimated_cases_state, how='left', left_index=True, right_index=True)

state_consolidated_df
state_consolidated_df
def plot_state_data(State):

    fig = make_subplots(rows=6, cols=1,shared_xaxes=False, 

                    subplot_titles=('Daily Confirmed Cases', 'Accumulative Confirmed Cases', 'Daily Deaths', 'Accumulative Deaths', 'Estimated Mortality Rate', 'Estimated Current Cases'))

    fig.add_trace(go.Bar(x=state_consolidated_df.loc[State].index,y=state_consolidated_df.loc[State, 'Daily New Cases'],

                    name='Daily Confirmed Cases'),

                    row=1, col=1)

    fig.add_trace(go.Scatter(x=state_consolidated_df.loc[State].index,y=state_consolidated_df.loc[State, 'Total Confirmed Cases'],

                    name='Accumulative Confirmed Cases',

                    line=dict(color='firebrick', width=3, dash='dot')),

                    row=2, col=1)

    fig.add_trace(go.Bar(x=state_consolidated_df.loc[State].index,y=state_consolidated_df.loc[State,'Daily New Deaths'],

                    name='Daily Deaths'),

                    row=3, col=1)

    fig.add_trace(go.Scatter(x=state_consolidated_df.loc[State].index,y=state_consolidated_df.loc[State,'Total Deaths'],

                    name='Acculative Deaths',

                    line=dict(color='firebrick', width=3, dash='dot')),

                    row=4, col=1)

    fig.add_trace(go.Scatter(x=state_consolidated_df.loc[State].index,y=state_consolidated_df.loc[State,'Death to Cases Ratio'],

                    mode='lines+markers',

                    name='Estimated Mortality Rate',

                    line=dict(color='LightSkyBlue',width=2)),

                    row=5,col=1)

    fig.add_trace(go.Bar(x=state_consolidated_df.loc[State].index,y=state_consolidated_df.loc[State,'Estimated Current Cases'],

                    name='Estimated Current Cases'),

                    row=6,col=1)

    fig.append_trace(go.Bar(x=state_consolidated_df.loc[State].index,y=state_consolidated_df.loc[State,'Estimated Past Cases'],

                    name='Estimated Current Cases'),

                    row=6,col=1)

    

    fig.update_layout(height=1200, width=800,showlegend=False)

    return fig





StateList=['California','Oregon','New York','Texas','Florida','Utah','Wisconsin','Washington']

interact(plot_state_data, State=widgets.Dropdown(options=StateList))
mean_death_to_cases_ratio = state_consolidated_df.groupby('Province/State', as_index=True)['Death to Cases Ratio'].agg([np.mean])

mean_death_to_cases_ratio = mean_death_to_cases_ratio.dropna()

mean_death_to_cases_ratio = mean_death_to_cases_ratio[mean_death_to_cases_ratio['mean']>0]

mean_death_to_cases_ratio = mean_death_to_cases_ratio[mean_death_to_cases_ratio['mean']<0.06]
#Table showing mortality rate for each state or province

temp = mean_death_to_cases_ratio

temp_sorted = temp.sort_values(by=['mean'], ascending=False)

temp_sorted.style.background_gradient()
histogram_x = mean_death_to_cases_ratio['mean']

plt.hist(histogram_x, bins='auto')
mean_death_to_cases_ratio.mean()
#Function for the regression model

def concentration_equation(time, m, k):

    C_ao=0

    return ( C_ao**(1/m) + (1/m)*k*time)**(m)



#Regression model. Place is either a country or state, examples "US", "New York"

#group is either 'Country/Region' or 'Province/State' 

#col_name is the column that you are running the 

#model on.  For instance "Total Confirmed Cases" or "Total Deaths", df is the dataframe.  This could either be

#the country or the state dataframe. 

#Starting date is a number representing the day at which you want the regression to start. For example, 

#if cases didn't really start occuring until day 30 in a specific area, then this number would be 30. The model

#will cut the dataframe to only includes dates after day 30. 

def regression_model(place, group, col_name, df, starting_date):

    

    #arrange the data into either country or state groups

    group= df.groupby(group)

    group = group.get_group(place)



    #cut the data up to the date that you want to start the regression model. 

    group= group.iloc[starting_date:]



    ydata=group.loc[place, col_name]



    time_data = (group.loc[place].index - group.loc[place].index[0]).days

    

    #reshape the data to 1D arrays for curve_fit

    ydata= ydata.values.reshape(-1, 1)

    time_data=time_data.values.reshape(-1,1)

    

    ydata = ydata.flatten()

    time_data= time_data.flatten()

    

    popt, pcov = curve_fit(concentration_equation, time_data, ydata, maxfev=5000)

    

    return popt



    
us_consolidated_cases_df = country_consolidated_df.groupby('Country/Region')

us_consolidated_cases_df = us_consolidated_cases_df.get_group('US')

us_consolidated_cases_df =  us_consolidated_cases_df[(us_consolidated_cases_df.loc['US'].index > '2020-02-20') & (us_consolidated_cases_df.loc['US'].index <= '2020-03-28')]





#date_list = country_consolidated_df.loc['US'].index



us_consolidated_cases_df


date_list_predicted = pd.date_range(start="2020-02-23", end="2020-04-15")

len(date_list_predicted)
#Model for cases in US

US_case_parameters = regression_model('US','Country/Region','Total Confirmed Cases', us_consolidated_cases_df, 0)



#rerun model to predict for dates March 30 - April 29. I'm doing this by running the model for an extra month.

x_time_predict = np.arange(1, len(date_list_predicted)+1, 1)

y_predict_cases_US = concentration_equation(x_time_predict, US_case_parameters[0], US_case_parameters[1])



#Model for deaths in US

US_death_parameters = regression_model('US','Country/Region','Total Deaths', us_consolidated_cases_df, 0)



#rerun model to predict for dates March 30 - April 29. I'm doing this by running the model for an extra month.

x_time_predict = np.arange(1, len(date_list_predicted)+1, 1)

y_predict_deaths_US = concentration_equation(x_time_predict, US_death_parameters[0], US_death_parameters[1])



#Create plots

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))



ax1.plot(us_consolidated_cases_df.loc['US'].index, us_consolidated_cases_df['Total Confirmed Cases'],'o')

ax1.plot(date_list_predicted, y_predict_cases_US )

plt.xlabel('Days')

plt.ylabel('Model and Predicted Cases')

plt.title('US - Model and Predicted Cases')



ax2.plot(us_consolidated_cases_df.loc['US'].index, us_consolidated_cases_df['Total Deaths'],'o')

ax2.plot(date_list_predicted, y_predict_deaths_US )         

plt.xlabel('Days')

plt.ylabel('Model and Predicted Deaths')

plt.title('US - Model and Predicted Deaths')         





         
italy_consolidated_cases_df = country_consolidated_df.groupby('Country/Region')

italy_consolidated_cases_df = italy_consolidated_cases_df.get_group('Italy')

italy_consolidated_cases_df =  italy_consolidated_cases_df[(italy_consolidated_cases_df.loc['Italy'].index > '2020-02-20') & (italy_consolidated_cases_df.loc['Italy'].index <= '2020-03-28')]





#date_list = country_consolidated_df.loc['US'].index



italy_consolidated_cases_df
date_list_predicted = pd.date_range(start="2020-02-23", end="2020-04-15")

len(date_list_predicted)
#Model for cases in US

italy_case_parameters = regression_model('Italy','Country/Region','Total Confirmed Cases', italy_consolidated_cases_df, 0)



#rerun model to predict for dates March 30 - April 29. I'm doing this by running the model for an extra month.

x_time_predict = np.arange(1, len(date_list_predicted)+1, 1)

y_predict_cases_italy = concentration_equation(x_time_predict, italy_case_parameters[0], italy_case_parameters[1])



#Model for deaths in US

italy_death_parameters = regression_model('Italy','Country/Region','Total Deaths', italy_consolidated_cases_df, 0)



#rerun model to predict for dates March 30 - April 29. I'm doing this by running the model for an extra month.

x_time_predict = np.arange(1, len(date_list_predicted)+1, 1)

y_predict_deaths_italy = concentration_equation(x_time_predict, italy_death_parameters[0], italy_death_parameters[1])



#Create plots

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))



ax1.plot(italy_consolidated_cases_df.loc['Italy'].index, italy_consolidated_cases_df['Total Confirmed Cases'],'o')

ax1.plot(date_list_predicted, y_predict_cases_italy )

plt.xlabel('Days')

plt.ylabel('Model and Predicted Cases')

plt.title('Italy - Model and Predicted Cases')



ax2.plot(italy_consolidated_cases_df.loc['Italy'].index, italy_consolidated_cases_df['Total Deaths'],'o')

ax2.plot(date_list_predicted, y_predict_deaths_italy )         

plt.xlabel('Days')

plt.ylabel('Model and Predicted Deaths')

plt.title('Italy - Model and Predicted Deaths')         



spain_consolidated_cases_df = country_consolidated_df.groupby('Country/Region')

spain_consolidated_cases_df = spain_consolidated_cases_df.get_group('Spain')

spain_consolidated_cases_df =  spain_consolidated_cases_df[(spain_consolidated_cases_df.loc['Spain'].index > '2020-02-20') & (spain_consolidated_cases_df.loc['Spain'].index <= '2020-03-28')]





#date_list = country_consolidated_df.loc['US'].index



spain_consolidated_cases_df
date_list_predicted = pd.date_range(start="2020-02-23", end="2020-04-15")

len(date_list_predicted)
#Model for cases in US

spain_case_parameters = regression_model('Spain','Country/Region','Total Confirmed Cases', spain_consolidated_cases_df, 0)



#rerun model to predict for dates March 30 - April 29. I'm doing this by running the model for an extra month.

x_time_predict = np.arange(1, len(date_list_predicted)+1, 1)

y_predict_cases_spain = concentration_equation(x_time_predict, spain_case_parameters[0], spain_case_parameters[1])



#Model for deaths in US

spain_death_parameters = regression_model('Spain','Country/Region','Total Deaths', spain_consolidated_cases_df, 0)



#rerun model to predict for dates March 30 - April 29. I'm doing this by running the model for an extra month.

x_time_predict = np.arange(1, len(date_list_predicted)+1, 1)

y_predict_deaths_spain = concentration_equation(x_time_predict, spain_death_parameters[0], spain_death_parameters[1])



#Create plots

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))



ax1.plot(spain_consolidated_cases_df.loc['Spain'].index, spain_consolidated_cases_df['Total Confirmed Cases'],'o')

ax1.plot(date_list_predicted, y_predict_cases_spain )

plt.xlabel('Days')

plt.ylabel('Model and Predicted Cases')

plt.title('Spain - Model and Predicted Cases')



ax2.plot(spain_consolidated_cases_df.loc['Spain'].index, spain_consolidated_cases_df['Total Deaths'],'o')

ax2.plot(date_list_predicted, y_predict_deaths_spain )         

plt.xlabel('Days')

plt.ylabel('Model and Predicted Deaths')

plt.title('Spain - Model and Predicted Deaths')         
southkorea_consolidated_cases_df = country_consolidated_df.copy()



parameters = regression_model('Korea, South','Country/Region','Total Confirmed Cases', southkorea_consolidated_cases_df, 30)

southkorea_consolidated_deaths_df = country_consolidated_df.copy()



parameters = regression_model('Korea, South','Country/Region','Total Deaths', southkorea_consolidated_deaths_df, 30)

ny_consolidated_cases_df = state_consolidated_df.groupby('Province/State')

ny_consolidated_cases_df = ny_consolidated_cases_df.get_group('New York')

ny_consolidated_cases_df =  ny_consolidated_cases_df[(ny_consolidated_cases_df.loc['New York'].index > '2020-03-7') & (ny_consolidated_cases_df.loc['New York'].index <= '2020-03-28')]





#date_list = country_consolidated_df.loc['US'].index



ny_consolidated_cases_df
date_list_predicted = pd.date_range(start="2020-03-09", end="2020-04-15")

len(date_list_predicted)
#Model for cases in US

ny_case_parameters = regression_model('New York','Province/State','Total Confirmed Cases', ny_consolidated_cases_df, 0)



#rerun model to predict for dates March 30 - April 29. I'm doing this by running the model for an extra month.

x_time_predict = np.arange(1, len(date_list_predicted)+1, 1)

y_predict_cases_ny = concentration_equation(x_time_predict, ny_case_parameters[0], ny_case_parameters[1])



#Model for deaths in US

ny_death_parameters = regression_model('New York','Province/State','Total Deaths', ny_consolidated_cases_df, 0)



#rerun model to predict for dates March 30 - April 29. I'm doing this by running the model for an extra month.

x_time_predict = np.arange(1, len(date_list_predicted)+1, 1)

y_predict_deaths_ny = concentration_equation(x_time_predict, ny_death_parameters[0], ny_death_parameters[1])



#Create plots

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))



ax1.plot(ny_consolidated_cases_df.loc['New York'].index, ny_consolidated_cases_df['Total Confirmed Cases'],'o')

ax1.plot(date_list_predicted, y_predict_cases_ny )

plt.xlabel('Days')

plt.ylabel('Model and Predicted Cases')

plt.title('New York - Model and Predicted Cases')



ax2.plot(ny_consolidated_cases_df.loc['New York'].index, ny_consolidated_cases_df['Total Deaths'],'o')

ax2.plot(date_list_predicted, y_predict_deaths_ny )         

plt.xlabel('Days')

plt.ylabel('Model and Predicted Deaths')

plt.title('New York - Model and Predicted Deaths')         