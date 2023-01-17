import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

from statsmodels.tsa.stattools import adfuller

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

#reading the input dataset into the pandas df
covid_19_india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

#sorting by date as dayfirst
# Columns in original dataset
#'Sno', 'Date', 'Time', 'State/UnionTerritory','ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured',
#'Deaths', 'Confirmed'

#creating the workable dataframe from the Kaggle CSV. It is undisturbed and all dependent data sets are made from this
covid_19_india['Date'] = pd.to_datetime(covid_19_india['Date'], dayfirst=True)

#print(covid_19_india.columns)

#creating workable dataframe grouped by state and date. This is further used for plotting of trends per date
CS_covid_state = covid_19_india[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
CS_covid_state = CS_covid_state.groupby(['Date','State/UnionTerritory'])[['Confirmed', 'Cured','Deaths']].sum().reset_index()
#creating workable dataframe grouped by time
#columns of dataset 'Date', 'Confirmed', 'Cured', 'Deaths', 'Active', 'new_case/day','growth_ratio', 'mortality'

CS_covid_Time = covid_19_india[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
CS_covid_Time = CS_covid_Time.groupby('Date')[['Confirmed', 'Cured','Deaths']].sum().reset_index()
CS_covid_Time['Active'] = CS_covid_Time['Confirmed'] - CS_covid_Time['Cured'] - CS_covid_Time['Deaths']
CS_covid_Time['new_case/day'] = CS_covid_Time['Confirmed'] - CS_covid_Time['Confirmed'].shift(1)
CS_covid_Time['growth_ratio'] = CS_covid_Time['new_case/day'] / CS_covid_Time['new_case/day'].shift(1)
CS_covid_Time['new_case/day'] = CS_covid_Time['new_case/day'].replace(np.nan,'0')
CS_covid_Time['growth_ratio'] = CS_covid_Time['growth_ratio'].replace(np.nan,'0')
CS_covid_Time['growth_ratio'] = CS_covid_Time['growth_ratio'].replace(float('inf'),'0')

#print(CS_covid_Time)
# plotting the original DF.
CS_covid_melt_df = pd.melt(CS_covid_Time, id_vars=['Date'], value_vars=['Confirmed','Active','Cured','new_case/day','Deaths'])
target_date = CS_covid_melt_df['Date'].max()
fig = px.line(CS_covid_melt_df, x="Date", y="value", color='variable', 
              title=f'All-India Cases Over Time {target_date}')
fig.show()
#checking the stationarity of the timeseries by plotting the mean of the features. 
CS_covid_stationarity = CS_covid_Time.groupby('Date')[['Confirmed', 'Cured','Deaths']].sum().reset_index()

r= CS_covid_stationarity.rolling(window=100)
CS_covid_stationarity['Confirmed'].plot(color = 'gray',label='Confirmed')
r.mean()['Confirmed'].plot(color ='red',label='Confirmed Mean')
CS_covid_stationarity['Cured'].plot(color = 'black',label='Cured')
r.mean()['Cured'].plot(color ='green',label='Cured Mean')
CS_covid_stationarity['Deaths'].plot(color = 'blue',label='Deaths')
r.mean()['Deaths'].plot(color ='pink', label='Deaths Mean')


###########
CS_covid_melt_df = pd.melt(CS_covid_stationarity, id_vars=['Date'], value_vars=['Confirmed','Deaths','Cured'])
target_date = CS_covid_melt_df['Date'].max()
fig = px.line(CS_covid_melt_df, x="Date", y="value", color='variable', 
              title=f'All-India Cases Over Time Mean {target_date}')
#fig.show()
# As shown above, the date is non-stationary and thus require a log transformation.
# Following is done to achieve log transformed data

CS_covid_Time_Analysis = CS_covid_Time[['Date','Confirmed', 'Cured', 'Deaths']]

#### calculating log transformation

CS_covid_Time_Analysis['log_Cured'] = np.log(CS_covid_Time_Analysis['Cured'])
CS_covid_Time_Analysis['log_Deaths'] = np.log(CS_covid_Time_Analysis['Deaths'])
CS_covid_Time_Analysis['log_Confirmed'] = np.log(CS_covid_Time_Analysis['Confirmed'])

#CS_covid_Time_Analysis['log_Growth_Ratio'] = np.log((CS_covid_Time_Analysis['growth_ratio']!=0))
#CS_covid_Time_Analysis['log_New_Case/Day'] = np.log((CS_covid_Time_Analysis['new_case/day']!=0))

##### calculating sqrt transformation.
# but the sqrt transformation is not helpful in this case. So no need to use it.

#CS_covid_Time_Analysis = covid_19_india[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
#CS_covid_Time_Analysis['Sqrt_Cured'] = CS_covid_Time_Analysis['Cured']**1/2
#CS_covid_Time_Analysis['Sqrt_Deaths'] = CS_covid_Time_Analysis['Deaths']**1/2
#CS_covid_Time_Analysis['Sqrt_Confirmed'] = CS_covid_Time_Analysis['Confirmed']**1/2

#CS_covid_Time_Analysis = CS_covid_Time_Analysis.groupby('Date')[['Confirmed', 'Cured','Deaths','Sqrt_Cured','Sqrt_Deaths','Sqrt_Confirmed']].sum().reset_index()

#print(CS_covid_Time_Analysis)
#print(CS_covid_Time)
# plotting the original features and log features to show that transformation has helped

CS_covid_melt_df_analysis = pd.melt(CS_covid_Time_Analysis, id_vars=['Date'], value_vars=['Confirmed','Cured','Deaths'])
target_date = CS_covid_melt_df_analysis['Date'].max()
fig = px.line(CS_covid_melt_df_analysis, x="Date", y="value", color='variable', 
              title=f'All-India Cases Over Time {target_date}')
#fig.show()

CS_covid_melt_df_analysis = pd.melt(CS_covid_Time_Analysis, id_vars=['Date'], value_vars=['log_Confirmed','log_Cured','log_Deaths'])
target_date = CS_covid_melt_df_analysis['Date'].max()
fig = px.line(CS_covid_melt_df_analysis, x="Date", y="value", color='variable', 
              title=f'All-India Cases (log) Over Time {target_date}')
fig.show()
fig = px.line(CS_covid_melt_df, x="Date", y="value", color='variable',
              title="All-India Cases Over Time (Log scale)",
             log_y =True)
#fig.show()
# plotting the overall mortality rate across India and its trend with time
CS_covid_Time['mortality'] = CS_covid_Time['Deaths'] / CS_covid_Time['Confirmed']

fig = px.line(CS_covid_Time, x="Date", y="mortality", 
              title="All-India Mortality Rate Over Time")
fig.show()
# analysis of number of confirmed cases and finding out how many states have how many cases
target_date = CS_covid_state['Date'].max()

print('As per Date:', target_date)
for i in [1, 10, 100, 1000, 10000,25000,35000,70000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,450000,500000]:
    n_states = len(CS_covid_state.query('(Date == @target_date) & Confirmed > @i'))
    print(f'{n_states} states have more than {i} confirmed cases')
pio.templates.default = "plotly_dark"
# From the state DF, top_state_df is made which queries the dataframe and sort the output of query in descending order of number of confirmed cases
#Columns of DF - Date', 'State/UnionTerritory', 'Confirmed', 'Cured', 'Deaths','Active'

top_states_df = CS_covid_state.query('(Date == @target_date) & (Confirmed > 35000)').sort_values('Confirmed', ascending=False)
top_states_df['Active'] = top_states_df['Confirmed'] - top_states_df['Cured'] - top_states_df['Deaths']
top_states_melt_df = pd.melt(top_states_df, id_vars='State/UnionTerritory', value_vars=['Confirmed','Active', 'Cured','Deaths'])

fig = px.bar(top_states_melt_df.iloc[::-1],
             x='value', y='State/UnionTerritory', color='variable', barmode='group',
             title=f'Confirmed/Cured/Deaths as on {target_date}', text='value', height=500, orientation='h')
fig.show()
# From the state DF, calculating the overall mortality rate across India
top_state_df = CS_covid_state.query('(Date == @target_date) & (Confirmed > 35000)')
top_state_df['mortality_rate'] = CS_covid_state['Deaths'] / CS_covid_state['Confirmed']
top_state_df = top_state_df.sort_values('mortality_rate', ascending=False)
print(top_state_df.shape)

fig = px.bar(top_state_df[:].iloc[::-1],
             x='mortality_rate', y='State/UnionTerritory',
             title=f'Mortality rate HIGH as on {target_date}', text='mortality_rate', height=500, orientation='h',color='mortality_rate')
fig.show()
# arranging the data of mortality rate in descending order to find out the states with lowest mortality
fig = px.bar(top_state_df[-15:].iloc[::-1],
             x='mortality_rate', y='State/UnionTerritory',
             title=f'Fifteen Lowest Mortality rate states on {target_date}', text='mortality_rate', height=500, orientation='h',color='mortality_rate')
fig.show()
#From the DF containing state level information, taking out information of Gujarat state only and saving it for checking the trends
#Columns of DF - 'Date', 'State/UnionTerritory', 'Confirmed', 'Cured', 'Deaths','prev_confirmed', 'new_case', 'prev_new_case', 'growth_factor','prev_cured', 'cured_case_per_day', 'active'

covid_19_state_ = CS_covid_state.query('Date > "2020-01-01"')

covid_19_state_['prev_confirmed'] = covid_19_state_.groupby('State/UnionTerritory')['Confirmed'].shift(1)
covid_19_state_['new_case'] = covid_19_state_['Confirmed'] - covid_19_state_['prev_confirmed']
covid_19_state_['new_case'].fillna(0, inplace=True)

covid_19_state_['prev_new_case'] = covid_19_state_.groupby('State/UnionTerritory')['new_case'].shift(1)
covid_19_state_['growth_factor'] = covid_19_state_['new_case'] / covid_19_state_['prev_new_case']
covid_19_state_['growth_factor'].fillna(0, inplace=True)
covid_19_state_['growth_factor'] = covid_19_state_['growth_factor'].replace(float('inf'),'0')

covid_19_state_['prev_cured'] = covid_19_state_.groupby('State/UnionTerritory')['Cured'].shift(1)
covid_19_state_['cured_case_per_day'] = covid_19_state_['Cured'] - covid_19_state_['prev_cured']

covid_19_state_['active'] = covid_19_state_['Confirmed'] - covid_19_state_['Cured'] - covid_19_state_['Deaths']

covid_19_state_Gujarat = covid_19_state_[covid_19_state_['State/UnionTerritory']=='Gujarat']
#print(covid_19_state_Gujarat)

covid_19_state_Karnataka = covid_19_state_[covid_19_state_['State/UnionTerritory']=='Karnataka']

covid_19_state_Uttar_Pradesh = covid_19_state_[covid_19_state_['State/UnionTerritory']=='Uttar Pradesh']
def state_wise_patients(name,df):
    data = df.loc[df['State/UnionTerritory']==name]
    df = data[['Confirmed','Cured','new_case','growth_factor','Date','State/UnionTerritory','active']]
#     data = df.groupby('Date')['Confirmed'].nunique()
    data = data.reset_index()
    data['Date']=pd.to_datetime(data['Date'],format = '%d/%m/%Y')
    data = data.sort_values(by=['Date'], ascending=True)
#     data['id'] = data.id.cumsum()
    return data
collection = {}
covid_19_states_for_analysis = ['Gujarat','Karnataka','Uttar Pradesh']
#for i in covid_19_state_['State/UnionTerritory'].unique():
for i in covid_19_states_for_analysis:
    collection['Patients in '+ str(i)] = state_wise_patients(i,covid_19_state_)

#collection['Patients in Gujrat'] = state_wise_patients('Gujarat',covid_19_state_)
#print(collection)
keys = list(collection.keys())
print(keys)
visible_True=[]
for i in range(len(keys)):
    visible_True.append(True)
def t2f(i):
    visible = []
    for a in range(len(keys)):
        if a == i:
            visible.append(True)
        else:
            visible.append(False)
    return visible
def create_buttons(keys,title):
    l=[dict(label = 'All',
                  method = 'update',
                  args = [{'visible': visible_True},
                          {'title': title+' India',
                           'showlegend':True}])]
    for i in range(len(keys)):
        l.append(dict(label = keys[i],
                  method = 'update',
                  args = [{'visible': t2f(i)}, # the index of True aligns with the indices of plot traces
                          {'title': title+keys[i].split('in')[1],
                           'showlegend':True}]))
    return l
# Plotting trend for states
fig = go.Figure()
keys = list(collection.keys())
i=0
for column in collection:
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].growth_factor,
            name = keys[i] + ' Growth Factor'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].new_case/100,
            name = keys[i] + ' New Cases/100'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].active/1000,
            name = keys[i] + ' Active Cases/1000'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].Deaths/1000,
            name = keys[i] + ' Deaths/1000'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].cured_case_per_day/1000,
            name = keys[i] + ' Cured Cases per day/1000'
        )
    )
    fig.add_trace(
        go.Line(
            x = collection[column].Date,
            y = collection[column].Confirmed/1000,
            name = keys[i] + ' Confirmed Cases/1000'
        )
    )
    i = i+1
   
fig.update_layout(updatemenus=[go.layout.Updatemenu( active=0,buttons=list(create_buttons(keys,'Confirmed Cases:')))])
fig.show()
#another way of plotting trend for Gujarat
covid_19_GJ_melt = pd.melt(covid_19_state_Gujarat, id_vars=['Date'], value_vars=['Confirmed','active', 'Cured','Deaths','cured_case_per_day','growth_factor'])
target_date = covid_19_GJ_melt['Date'].max()

fig = px.line(covid_19_GJ_melt, x="Date", y="value", color='variable', 
              title=f'Trend Chart for Gujarat as on {target_date}')
    
fig.show()
covid_19_state_Gujarat.corr()
#another way of plotting trend for Karnataka
covid_19_KA_melt = pd.melt(covid_19_state_Karnataka, id_vars=['Date'], value_vars=['Confirmed','active', 'Cured','Deaths','cured_case_per_day','growth_factor'])
target_date = covid_19_KA_melt['Date'].max()

fig = px.line(covid_19_KA_melt, x="Date", y="value", color='variable', 
              title=f'Trend Chart for Karnataka as on {target_date}')
    
fig.show()
covid_19_state_Karnataka.corr()
#another way of plotting trend for UP
covid_19_UP_melt = pd.melt(covid_19_state_Uttar_Pradesh, id_vars=['Date'], value_vars=['Confirmed','active', 'Cured','Deaths','cured_case_per_day','growth_factor'])
target_date = covid_19_UP_melt['Date'].max()

fig = px.line(covid_19_UP_melt, x="Date", y="value", color='variable', 
              title=f'Trend Chart for Karnataka as on {target_date}')
    
fig.show()
covid_19_state_Uttar_Pradesh.corr()
df_india = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv",index_col= 0)
df_india['Date'] = pd.to_datetime(df_india['Date'], dayfirst=True)
state_report = df_india.groupby(by = 'State/UnionTerritory').max().reset_index()
import IPython
IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1977187" data-url="https://flo.uri.sh/visualisation/1977187/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
df_india.dtypes
latest = df_india[df_india['Date'] > pd.to_datetime('2020-04-01')]

latest2 = latest.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured',"Date"].max().reset_index()

latest2['Active'] = latest2['Confirmed'] - (latest2['Deaths'] - latest2['Cured'])

state_list = list(latest2.sort_values('Active',ascending = False)['State/UnionTerritory'])[0:15]

states_confirmed = {}
states_deaths = {}
states_recovered = {}
states_active = {}
states_dates = {} 
# Columns in original dataset
# 'Sno', 'State/UT', 'NumPrimaryHealthCenters_HMIS','NumCommunityHealthCenters_HMIS', 'NumSubDistrictHospitals_HMIS',
# 'NumDistrictHospitals_HMIS', 'TotalPublicHealthFacilities_HMIS','NumPublicBeds_HMIS', 'NumRuralHospitals_NHP18', 'NumRuralBeds_NHP18',
# 'NumUrbanHospitals_NHP18', 'NumUrbanBeds_NHP18']

#reading the CSV from Kaggle and storing to DF

CS_covid_hosp_beds = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')

#CS_covid_hosp_beds = CS_covid_hosp_beds.loc[CS_covid_hosp_beds['State/UT']=='Gujarat']

#adding a column for total no of beds
CS_covid_hosp_beds['NumPrimaryHealthCenters_HMIS'] = CS_covid_hosp_beds['NumPrimaryHealthCenters_HMIS'].replace(",",'',inplace=True)

CS_covid_hosp_beds['NumSubDistrictHospitals_HMIS'] = CS_covid_hosp_beds['NumSubDistrictHospitals_HMIS'].replace(np.nan,0)
CS_covid_hosp_beds['NumPrimaryHealthCenters_HMIS'] = CS_covid_hosp_beds['NumPrimaryHealthCenters_HMIS'].replace(np.nan,0)

CS_covid_hosp_beds['NumPrimaryHealthCenters_HMIS'] = pd.to_numeric(CS_covid_hosp_beds['NumPrimaryHealthCenters_HMIS'])
CS_covid_hosp_beds['NumSubDistrictHospitals_HMIS'] = pd.to_numeric(CS_covid_hosp_beds['NumSubDistrictHospitals_HMIS'])

CS_covid_hosp_beds['TotalBeds'] = CS_covid_hosp_beds['NumPrimaryHealthCenters_HMIS']+CS_covid_hosp_beds['NumSubDistrictHospitals_HMIS']+CS_covid_hosp_beds['NumCommunityHealthCenters_HMIS']+CS_covid_hosp_beds['NumDistrictHospitals_HMIS']+CS_covid_hosp_beds['TotalPublicHealthFacilities_HMIS']+CS_covid_hosp_beds['NumPublicBeds_HMIS']+CS_covid_hosp_beds['NumRuralHospitals_NHP18'] + CS_covid_hosp_beds['NumRuralBeds_NHP18']+CS_covid_hosp_beds['NumUrbanHospitals_NHP18']+CS_covid_hosp_beds['NumUrbanBeds_NHP18']

#CS_covid_hosp_beds['TotalBeds'] = CS_covid_hosp_beds[['NumPrimaryHealthCenters_HMIS','NumCommunityHealthCenters_HMIS','NumSubDistrictHospitals_HMIS','NumDistrictHospitals_HMIS','TotalPublicHealthFacilities_HMIS','NumPublicBeds_HMIS','NumRuralHospitals_NHP18','NumRuralBeds_NHP18','NumUrbanHospitals_NHP18','NumUrbanBeds_NHP18']].sum()
#CS_covid_hosp_beds.describe()

#print(CS_covid_hosp_beds)
pio.templates.default = "plotly_white"
CS_covid_hosp_beds = CS_covid_hosp_beds.sort_values('TotalBeds',ascending=True)

fig = px.bar(CS_covid_hosp_beds,
             x='TotalBeds', y='State/UT',
             title=f'Total Number of Beds/States as of {target_date}', height=1000, orientation='h',color_continuous_scale='Inferno',color='TotalBeds')
fig.show()
from datetime import timedelta
import datetime
# Reading the CSV from Kaggle and storing to DF with details of no of tests done
#validated data against https://www.statista.com/statistics/1107186/india-coronavirus-covid-19-testing-numbers-by-state/
#columns of DF - 'Date', 'State', 'TotalSamples', 'Negative', 'Positive','Perct_Positive', 'Perct_Negative'
CS_covid_test_labs = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')

#Columns of DF - 'Date', 'State', 'TotalSamples', 'Negative', 'Positive'
CS_covid_test_all_states = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')

#information of Gujarat state will be captured in CS_covid_test_labs
CS_covid_test_labs = CS_covid_test_labs[CS_covid_test_labs["State"] == 'Gujarat']

CS_covid_test_labs['Date'] = pd.to_datetime(CS_covid_test_labs['Date'], dayfirst=True)
CS_covid_test_labs['Negative'] = CS_covid_test_labs['Negative'].replace(np.nan,CS_covid_test_labs['TotalSamples']-CS_covid_test_labs['Positive'])
CS_covid_test_labs['Negative'] = pd.to_numeric(CS_covid_test_labs['Negative'])

CS_covid_test_labs['Perct_Positive'] = CS_covid_test_labs['Positive']/CS_covid_test_labs['TotalSamples']
CS_covid_test_labs['Perct_Negative'] = CS_covid_test_labs['Negative']/CS_covid_test_labs['TotalSamples']

# in the state level information collected, replacing the NaN values by the difference between Total samples and positive samples
CS_covid_test_all_states['Negative'] = CS_covid_test_all_states['Negative'].replace(np.nan,CS_covid_test_all_states['TotalSamples']-CS_covid_test_all_states['Positive'])

# Collecting the information and grouping on state
#as the data is till two days ago, the query is executed for that day
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

CS_covid_test_all_states['Date'] = pd.to_datetime(CS_covid_test_all_states['Date'], dayfirst=True)
CS_covid_test_all_states = CS_covid_test_all_states.query('Date == @yesterday').sort_values('TotalSamples', ascending=False)
#information of Karnataka state will be captured in CS_covid_test_labs
CS_covid_test_labs_k = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')
CS_covid_test_labs_k = CS_covid_test_labs_k[CS_covid_test_labs_k["State"] == 'Karnataka']

CS_covid_test_labs_k['Date'] = pd.to_datetime(CS_covid_test_labs_k['Date'], dayfirst=True)
CS_covid_test_labs_k['Negative'] = CS_covid_test_labs_k['Negative'].replace(np.nan,CS_covid_test_labs_k['TotalSamples']-CS_covid_test_labs_k['Positive'])
CS_covid_test_labs_k['Negative'] = pd.to_numeric(CS_covid_test_labs_k['Negative'])

CS_covid_test_labs_k['Perct_Positive'] = CS_covid_test_labs_k['Positive']/CS_covid_test_labs_k['TotalSamples']
CS_covid_test_labs_k['Perct_Negative'] = CS_covid_test_labs_k['Negative']/CS_covid_test_labs_k['TotalSamples']
#information of UP state will be captured in CS_covid_test_labs
CS_covid_test_labs_up = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')
CS_covid_test_labs_up = CS_covid_test_labs_up[CS_covid_test_labs_up["State"] == 'Uttar Pradesh']

CS_covid_test_labs_up['Date'] = pd.to_datetime(CS_covid_test_labs_up['Date'], dayfirst=True)
CS_covid_test_labs_up['Negative'] = CS_covid_test_labs_up['Negative'].replace(np.nan,CS_covid_test_labs_up['TotalSamples']-CS_covid_test_labs_up['Positive'])
CS_covid_test_labs_up['Negative'] = pd.to_numeric(CS_covid_test_labs_up['Negative'])

CS_covid_test_labs_up['Perct_Positive'] = CS_covid_test_labs_up['Positive']/CS_covid_test_labs_up['TotalSamples']
CS_covid_test_labs_up['Perct_Negative'] = CS_covid_test_labs_up['Negative']/CS_covid_test_labs_up['TotalSamples']
pio.templates.default = "plotly_white"
#plotting the tests DF
covid_19_GJ_test_melt = pd.melt(CS_covid_test_labs, id_vars=['Date'], value_vars=['TotalSamples','Negative','Positive'])
target_date = covid_19_GJ_test_melt['Date'].max()
fig = px.line(covid_19_GJ_test_melt, x="Date", y="value", color='variable', 
              title=f'Number of Tests done in Gujarat as on {target_date}')

fig.show()

covid_19_GJ_test_melt = pd.melt(CS_covid_test_labs, id_vars=['Date'], value_vars=['Perct_Positive','Perct_Negative'])
target_date = covid_19_GJ_test_melt['Date'].max()
fig = px.line(covid_19_GJ_test_melt, x="Date", y="value", color='variable', 
              title=f'Percentage Positive/Negative Test Results in Gujarat as on {target_date}')

fig.show()
#plotting the tests DF
covid_19_KA_test_melt = pd.melt(CS_covid_test_labs_k, id_vars=['Date'], value_vars=['TotalSamples','Negative','Positive'])
target_date = covid_19_KA_test_melt['Date'].max()
fig = px.line(covid_19_KA_test_melt, x="Date", y="value", color='variable', 
              title=f'Number of Tests done in Karnataka as on {target_date}')

fig.show()

covid_19_KA_test_melt = pd.melt(CS_covid_test_labs_k, id_vars=['Date'], value_vars=['Perct_Positive','Perct_Negative'])
target_date = covid_19_KA_test_melt['Date'].max()
fig = px.line(covid_19_KA_test_melt, x="Date", y="value", color='variable', 
              title=f'Percentage Positive/Negative Test Results in Karnataka as on {target_date}')

fig.show()
#plotting the tests DF
covid_19_UP_test_melt = pd.melt(CS_covid_test_labs_up, id_vars=['Date'], value_vars=['TotalSamples','Negative','Positive'])
target_date = covid_19_UP_test_melt['Date'].max()
fig = px.line(covid_19_UP_test_melt, x="Date", y="value", color='variable', 
              title=f'Number of Tests done in Uttar Pradesh as on {target_date}')

fig.show()

covid_19_UP_test_melt = pd.melt(CS_covid_test_labs_up, id_vars=['Date'], value_vars=['Perct_Positive','Perct_Negative'])
target_date = covid_19_UP_test_melt['Date'].max()
fig = px.line(covid_19_UP_test_melt, x="Date", y="value", color='variable', 
              title=f'Percentage Positive/Negative Test Results in Uttar Pradesh as on {target_date}')

fig.show()
# Reading the CSV from Kaggle and storing to DF. It contains the individual details of patients
# Columns of DF - 'Sno', 'AgeGroup', 'TotalCases', 'Percentage'
CS_covid_age_group = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
CS_covid_age_group
fig = px.pie(CS_covid_age_group, values='TotalCases', names='AgeGroup', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
# Reading the CSV from Kaggle and storing to DF. It contains the individual details of patients
# Columns of DF - 'id', 'government_id', 'diagnosed_date', 'age', 'gender','detected_city', 'detected_district', 'detected_state', 'nationality',
#'current_status', 'status_change_date', 'notes', 'illness_duration'
CS_covid_individual_info = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

#print(CS_covid_individual_info.tail(20))

CS_covid_individual_info = CS_covid_individual_info.loc[CS_covid_individual_info['detected_state']=='Gujarat']
#id	government_id	diagnosed_date	age	gender	detected_city	detected_district	detected_state	nationality	current_status	status_change_date	notes
# Realigning the dataframe on time basis

CS_covid_individual_info['status_change_date'] = pd.to_datetime(CS_covid_individual_info['status_change_date'], dayfirst=True)
CS_covid_individual_info['diagnosed_date'] = pd.to_datetime(CS_covid_individual_info['diagnosed_date'], dayfirst=True)

#CS_covid_individual_info = CS_covid_individual_info[['diagnosed_date','age','gender','detected_state','current_status','status_change_date']]

CS_covid_individual_info['illness_duration'] = CS_covid_individual_info['status_change_date'] - CS_covid_individual_info['diagnosed_date']

#CS_covid_individual_info['active_cases'] = (CS_covid_individual_info['current_status'] == 'Recovered').count()
#CS_covid_state_info = CS_covid_individual_info['gender','age','detected_state','current_status','illness_duration']

#CS_covid_state_info = CS_covid_individual_info.groupby(['detected_state'])['diagnosed_date','gender','age'].count().reset_index()
#print(CS_covid_state_info)
print(CS_covid_individual_info.tail(5))
#collecting the details of each state into a workable DF
#values are validated against covid19india.org
#Columns of DF - State/UnionTerritory,Confirmed,Cured,Deaths
CS_covid_state_info = CS_covid_state.query('(Date == @target_date) & (Confirmed > 0)').sort_values('Confirmed', ascending=False)
#reading the CSV from Kaggle and storing into workable DF
#Columns of DF - Sno', 'State/UnionTerritory', 'Population', 'Rural population',
# 'Urban population', 'Area', 'Density', 'Gender Ratio'
# numbers are validated against https://www.covid19india.org/state/xxxx

covid_19_india_population = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')

#renaming so that megre of Dataframes can be done
covid_19_india_population = covid_19_india_population.rename(columns ={"State / Union Territory":"State/UnionTerritory"}) 
#covid_19_india_population = covid_19_india_population.drop(columns=['Sno'])

# merging the two dataframes and the final one would be CS_covid_state_info
# Columns of DF - 'Date', 'State/UnionTerritory', 'Confirmed', 'Cured', 'Deaths', 'Population', 'Rural population', 'Urban population', 'Area', 'Density',
# 'Gender Ratio', 'Cases/Population'

CS_covid_state_info = pd.merge(CS_covid_state_info, covid_19_india_population, on="State/UnionTerritory")
CS_covid_state_info['Cases/Population'] = (CS_covid_state_info['Confirmed']/CS_covid_state_info['Population'])*1000000

# taking the population details in another DF for Gujarat
covid_19_Gujarat_population = CS_covid_state_info[CS_covid_state_info['State/UnionTerritory'] == 'Gujarat']

#covid_19_Gujarat_population
#covid_19_Gujarat_population

#CS_covid_state_info
#covid_19_Gujarat_population
pio.templates.default = "plotly_white"
CS_covid_state_info = CS_covid_state_info.sort_values('Cases/Population', ascending=True)

fig = px.bar(CS_covid_state_info,
             x='Cases/Population', y='State/UnionTerritory',
             title=f'Cases/1 Million Population {target_date}', height=1000, orientation='h',color_continuous_scale=px.colors.sequential.Cividis_r,color='Cases/Population')
fig.show()
CS_covid_state_info = CS_covid_state_info.sort_values('State/UnionTerritory', ascending=True)

fig = px.bar(CS_covid_state_info,
             x='Density', y='State/UnionTerritory',
             title=f'Population Density {target_date}', height=1000, orientation='h',color_continuous_scale=px.colors.sequential.Viridis, color='State/UnionTerritory')
fig.show()
import fbprophet
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#reading the input dataset into the pandas df
model_df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

#creating the workable dataframe from the Kaggle CSV. It is undisturbed and all dependent data sets are made from this
model_df['Date'] = pd.to_datetime(model_df['Date'], dayfirst=True)
#preparing the dataframe with only Cured, Confirmed and Death numbers
covid_19_india_model_df = model_df.groupby("Date")[["Cured","Deaths","Confirmed"]].sum().reset_index()
#covid_19_india_model_df.tail()
#Time Series Forecasting
covid_19_model = Prophet(interval_width = 0.95,growth ='linear',seasonality_mode='multiplicative',weekly_seasonality=False,daily_seasonality=False)



#covid_19_model.add_seasonality(name = "Monthly",period=180,fourier_order=5) #check after removing this as well
#separating dataset into confirmed and recovered data sets
covid_19_model_confirmed = covid_19_india_model_df[["Date","Confirmed"]]
covid_19_model_recovered = covid_19_india_model_df[["Date","Cured"]]
covid_19_model_deaths = covid_19_india_model_df[["Date","Deaths"]]

covid_19_model_confirmed.rename(columns={"Date":"ds","Confirmed":"y"},inplace=True)
# log-transform y
covid_19_model_confirmed['y'] = np.log(covid_19_model_confirmed['y'])

covid_19_model_recovered.rename(columns={"Date":"ds","Cured":"y"},inplace=True) 
# log-transform y
covid_19_model_recovered['y'] = np.log(covid_19_model_recovered['y'])

covid_19_model_deaths.rename(columns={"Date":"ds","Deaths":"y"},inplace=True)
# log-transform y
covid_19_model_deaths['y'] = np.log(covid_19_model_deaths['y'])

#taking the train and test sets
covid_19_train_set = covid_19_model_confirmed[:168]
covid_19_test_set = covid_19_model_confirmed[169:]
#model fit
covid_19_model.fit(covid_19_model_confirmed)


#covid_19_model.fit(covid_19_train_set)
#Calculating the future dates
covid_19_future_dates = covid_19_model.make_future_dataframe(periods=6, freq = 'm')

#covid_19_future_dates = covid_19_model.make_future_dataframe(periods=len(covid_19_test_set)*30,freq ='d')
covid_19_future_dates.tail()
#predictions 
covid_19_predictions = covid_19_model.predict(covid_19_future_dates)
covid_19_predictions.tail()
#plotting the predictions
covid_19_model.plot(covid_19_predictions)
#plotting the components
covid_19_model.plot_components(covid_19_predictions)
fig = covid_19_model.plot(covid_19_predictions)
var = add_changepoints_to_plot(fig.gca(),covid_19_model,covid_19_predictions)
# Cross Validation
covid_19_cv_df = cross_validation(covid_19_model,initial ='10 days', period='15 days', horizon = '120 days')
#covid_19_cv_df.tail()
#performance metrics
performance_metrics_results = performance_metrics(covid_19_cv_df)
print(performance_metrics_results)
#plotting the cross validation metrics - MAPE
fig = plot_cross_validation_metric(covid_19_cv_df, metric='mape')
#merging the predicted values and the original one
metric_df = covid_19_predictions.set_index('ds')[['yhat']].join(covid_19_model_confirmed.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)


#metric_df = covid_19_predictions.set_index('ds')[['yhat']].join(covid_19_train_set.set_index('ds').y).reset_index()
#metric_df.dropna(inplace=True)
# Checking the accuracy of the model
print("R2 SCORE")
print(r2_score(metric_df.y, metric_df.yhat))

print("Mean Squared Error")
print(mean_squared_error(metric_df.y, metric_df.yhat))

print("Mean Absolute Error")
print(mean_absolute_error(metric_df.y, metric_df.yhat))
#Time Series Forecasting
covid_19_model_r = Prophet(interval_width = 0.95,seasonality_mode='multiplicative',weekly_seasonality=False,daily_seasonality=False)
covid_19_model_recovered.tail()
covid_19_model_recovered['y'] = covid_19_model_recovered['y'].replace(float('-inf'),'0')
#model fit
covid_19_model_r.fit(covid_19_model_recovered)
#predictions 
covid_19_predictions_r = covid_19_model_r.predict(covid_19_future_dates)
#covid_19_predictions_r[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#plotting the predictions
covid_19_model_r.plot(covid_19_predictions_r)
#plotting the components
covid_19_model_r.plot_components(covid_19_predictions_r)
fig = covid_19_model_r.plot(covid_19_predictions_r)
var = add_changepoints_to_plot(fig.gca(),covid_19_model_r,covid_19_predictions_r)
#merging the predicted values and the original one
metric_df_r = covid_19_predictions_r.set_index('ds')[['yhat']].join(covid_19_model_recovered.set_index('ds').y).reset_index()
metric_df_r.dropna(inplace=True)
# Checking the accuracy of the model
print("R2 SCORE")
print(r2_score(metric_df_r.y, metric_df_r.yhat))

print("Mean Squared Error")
print(mean_squared_error(metric_df_r.y, metric_df_r.yhat))

print("Mean Absolute Error")
print(mean_absolute_error(metric_df_r.y, metric_df_r.yhat))
#Selecting the dataframe till 23rd March
lockdown1 = covid_19_model_confirmed[:52]
#Time Series Forecasting
covid_19_model_lk1 = Prophet(interval_width = 0.95,seasonality_mode='multiplicative',weekly_seasonality=False,daily_seasonality=False)
#Model
covid_19_model_lk1.fit(lockdown1)
#predictions 
covid_19_predictions_lk1 = covid_19_model_lk1.predict(covid_19_future_dates)
#plotting the predictions
covid_19_model_lk1.plot(covid_19_predictions_lk1)
covid_19_model.plot(covid_19_predictions)
#Selecting the dataframe till 14th April
lockdown2 = covid_19_model_confirmed[:76]
#lockdown2.tail()
#Time Series Forecasting
covid_19_model_lk2 = Prophet(interval_width = 0.95,seasonality_mode='multiplicative',weekly_seasonality=False,daily_seasonality=False)
#Model
covid_19_model_lk2.fit(lockdown2)
#predictions 
covid_19_predictions_lk2 = covid_19_model_lk2.predict(covid_19_future_dates)
#plotting the predictions
covid_19_model_lk2.plot(covid_19_predictions_lk2)
covid_19_model.plot(covid_19_predictions)
#Selecting the dataframe till 3rd May
lockdown3 = covid_19_model_confirmed[:95]
#lockdown3.tail()
#Time Series Forecasting
covid_19_model_lk3 = Prophet(interval_width = 0.95,seasonality_mode='multiplicative',weekly_seasonality=False,daily_seasonality=False)
#Model
covid_19_model_lk3.fit(lockdown3)
#predictions 
covid_19_predictions_lk3 = covid_19_model_lk3.predict(covid_19_future_dates)
#plotting the predictions
covid_19_model_lk3.plot(covid_19_predictions_lk3)
covid_19_model.plot(covid_19_predictions)
#Selecting the dataframe till 17th May
lockdown4 = covid_19_model_confirmed[:109]
#lockdown4.tail()
#Time Series Forecasting
covid_19_model_lk4 = Prophet(interval_width = 0.95,seasonality_mode='multiplicative',weekly_seasonality=False,daily_seasonality=False)
#Model
covid_19_model_lk4.fit(lockdown4)
#predictions 
covid_19_predictions_lk4 = covid_19_model_lk4.predict(covid_19_future_dates)
#plotting the predictions
covid_19_model_lk4.plot(covid_19_predictions_lk4)
covid_19_model.plot(covid_19_predictions)
fig = covid_19_model_lk1.plot(covid_19_predictions_lk1)
var = add_changepoints_to_plot(fig.gca(),covid_19_model_lk1,covid_19_predictions_lk1)

fig = covid_19_model_lk2.plot(covid_19_predictions_lk2)
var = add_changepoints_to_plot(fig.gca(),covid_19_model_lk2,covid_19_predictions_lk2)

fig = covid_19_model_lk1.plot(covid_19_predictions_lk3)
var = add_changepoints_to_plot(fig.gca(),covid_19_model_lk3,covid_19_predictions_lk3)

fig = covid_19_model_lk1.plot(covid_19_predictions_lk4)
var = add_changepoints_to_plot(fig.gca(),covid_19_model_lk4,covid_19_predictions_lk4)
