# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json,requests

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objs as go

import seaborn as sns

covid_india=pd.read_csv('/kaggle/input/covid19-corona-virus-india-dataset/complete.csv')

covid_india.head()

covid_india.columns

india_data_json = requests.get('https://api.rootnet.in/covid19-in/unofficial/covid19india.org/statewise').json()

df_india = pd.io.json.json_normalize(india_data_json['data']['statewise'])#normalize json data to a table

#df_india = df_india.set_index("state")

df_india

covid19=pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

covid19.head()

novel_confirmed=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

novel_confirmed.head()

novel_recovered=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

novel_recovered.head()

novel_death=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

novel_death.head()

#Data Preprocessing

covid_india['Total']=covid_india['Total Confirmed cases']

covid_india['active']=covid_india['Total']-covid_india['Death']-covid_india['Cured/Discharged/Migrated']

covid_india['Mortality']=covid_india['Death']/covid_india['active']

covid_india['Recovery']=covid_india['Cured/Discharged/Migrated']/covid_india['active']

covid_india['Recovery_Rate']=covid_india['Recovery']*100

#Indian trend in past weeks

df_group=covid_india.groupby('Date').sum().reset_index()

df_group.columns

df_group['active']

plt.figure(figsize=(20,20))

plt.plot(df_group.Total,label='Rise in Case')

plt.plot(df_group.active,label='Active')

plt.plot(df_group['Cured/Discharged/Migrated'],label='Cured')

plt.plot(df_group.Death,label='Death')

plt.legend()

plt.xlabel('Date')

plt.ylabel('Cases')

#Checking current situation of India 

ac=df_india['active'].sum()

rvd=df_india['recovered'].sum()

dt=df_india['deaths'].sum()

fig = go.Figure(data=[go.Pie(labels=['Active Cases','Cured','Death'],

                             values= [ac,rvd,dt],hole =.3)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=['#263fa3', '#2fcc41','#cc3c2f'], line=dict(color='#FFFFFF', width=2)))

fig.update_layout(title_text='Current Situation in India',plot_bgcolor='rgb(275, 270, 273)')

fig.show()

#Preprocessing of data to check trends in various weeks

df_3=df_group[(df_group['Date']>='2020-02-23') & (df_group['Date']<='2020-02-29')]

df_4=df_group[(df_group['Date']>='2020-03-01') & (df_group['Date']<='2020-03-07')]

df_5=df_group[(df_group['Date']>='2020-03-08') & (df_group['Date']<='2020-03-14')]

df_6=df_group[(df_group['Date']>='2020-03-15') & (df_group['Date']<='2020-03-21')]

df_7=df_group[(df_group['Date']>='2020-03-22') & (df_group['Date']<='2020-03-28')]

df_8=df_group[(df_group['Date']>='2020-03-29') & (df_group['Date']<='2020-04-05')]

df_9=df_group[(df_group['Date']>='2020-04-06') & (df_group['Date']<='2020-04-11')]

df_10=df_group[(df_group['Date']>='2020-04-12') & (df_group['Date']<='2020-04-18')]

df_11=df_group[(df_group['Date']>='2020-04-19') & (df_group['Date']<='2020-04-25')]

df_12=df_group[(df_group['Date']>='2020-04-26')&(df_group['Date']<='2020-05-02')]

sum1=df_3['Total'].max()

sum2=df_4['Total'].max()

sum3=df_5['Total'].max()

sum4=df_6['Total'].max()

sum5=df_7['Total'].max()

sum6=df_8['Total Confirmed cases'].max()

sum7=df_9['Total Confirmed cases'].max()

sum8=df_10['Total Confirmed cases'].max()

sum9=df_11['Total Confirmed cases'].max()



total_sum=[sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9]

total_sum

date1=df_3['Date'].max()

date2=df_4['Date'].max()

date3=df_5['Date'].max()

date4=df_6['Date'].max()

date5=df_7['Date'].max()

date6=df_8['Date'].max()

date7=df_9['Date'].max()

date8=df_10['Date'].max()

date9=df_11['Date'].max()

total_date=[date1,date2,date3,date4,date5,date6,date7,date8,date9]

total_date

dates=[df_3['Date'],df_4['Date'],df_5['Date'],df_6['Date'],df_7['Date'],df_8['Date']]

dates

#Plotting weekly trends in log scale

plt.figure(figsize=(20,20))

plt.plot(total_date,total_sum,label='Trend in Cases')

plt.legend()

plt.yscale('log')

plt.xlabel('Date')

plt.ylabel('No of cases')

fig=go.Figure()

fig.add_trace(go.Scatter(x=total_date, y=total_sum,

                    mode='lines+markers',marker_color='blue',name='Total Cases'))

fig.update_layout(title_text='Trend of Weekly Coronavirus Cases in India',plot_bgcolor='rgb(275,270,273)',width=600, height=600)

fig.show()

#Trend in Various States

df_data_group=covid19[['Date','State/UnionTerritory','Confirmed','Cured','Deaths']]

df_data_active=df_data_group.groupby(['Date','State/UnionTerritory'])['Confirmed'].sum().reset_index()

df_data_active=df_data_active.sort_values('Confirmed',ascending=False)

df_data_active

fig=px.area(df_data_active,x='Date',y='Confirmed',color='State/UnionTerritory',title='Spread in States over time')

fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=700, height=600)

#Recovery Rate



fig=go.Figure()

fig.add_trace(go.Scatter(x=covid_india['Date'],y=covid_india['Recovery_Rate'],mode='lines+markers',marker_color='red',name='Total Cases'))

fig.update_layout(title_text='Trend of Recovery Cases in India',plot_bgcolor='rgb(275,270,273)',width=600, height=600)

fig.show()

#Mortality Rate

fig=go.Figure()

fig.add_trace(go.Scatter(x=covid_india['Date'],y=covid_india['Mortality'],mode='lines+markers',marker_color='red',name='Total Cases'))

fig.update_layout(title_text='Trend of Mortality Rate in India',plot_bgcolor='rgb(275,270,273)',width=600, height=600)

fig.show()

#Covid 19 State Wise Count

fig=go.Figure(data=[go.Bar(name='Active',y=df_india['active'],x=df_india['state']),

                   go.Bar(name='Recovered',y=df_india['recovered'],x=df_india['state']),

                   go.Bar(name='Death',y=df_india['deaths'],x=df_india['state'])])



fig.update_layout(barmode='stack', height=900)

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_layout(title_text='Active Cases,Deaths and Recovered',plot_bgcolor='rgb(275,270,273)',width=600, height=800)

fig.show()

#State wise confirmed Cases

fig=go.Figure(data=[go.Bar(name='Active',y=df_india['confirmed'],x=df_india['state'])])

fig.update_layout(barmode='stack', height=900)

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_layout(title_text='Active Cases',plot_bgcolor='rgb(275,270,273)',width=600, height=800)

fig.show()

novel_confirmed=novel_confirmed.drop(['Province/State','Lat','Long'],axis=1)

novel_confirmed.columns

novel_confirmed=novel_confirmed.rename(columns={'Country/Region':'Region'})

#List of countries to analyse and comparing India with Rest of World for Confirmed Cases

novel_confirmed

list_country=['Korea, South','Italy','Spain','India','Iran']

novel_conf=novel_confirmed[novel_confirmed['Region'].isin(list_country)]

novel_conf_tr=novel_conf.T

novel_conf_tr=novel_conf_tr.rename(columns={131:'India',133:'Iran',137:'Italy',143:'Korea, South',201:'Spain'})

novel_conf_tr=novel_conf_tr.drop(novel_conf_tr.index[0])#removing 1st row

novel_conf_tr

novel_conf_tr.index.name='Date'

novel_conf_tr.head()

novel_conf_tr.plot()

novel_recovered=novel_recovered.drop(['Province/State','Lat','Long'],axis=1)

novel_recovered.columns

novel_recovered=novel_recovered.rename(columns={'Country/Region':'Region'})

#List of countries to analyse and comparing India with Rest of World for Recovered Cases

novel_recovered

list_country=['Korea, South','Italy','Spain','India','Iran']

novel_reco=novel_recovered[novel_recovered['Region'].isin(list_country)]

novel_reco_tr=novel_reco.T

novel_reco_tr

#novel_reco_tr=novel_reco_tr.rename(columns={131:'India',133:'Iran',137:'Italy',143:'Korea, South',201:'Spain'})

#novel_reco_tr=novel_reco_tr.drop(novel_conf_tr.index[0])#removing 1st row

#novel_reco_tr

#novel_reco_tr.index.name='Date'

novel_reco_tr=novel_reco_tr.iloc[1:]

novel_reco_tr.head()

novel_reco_tr.plot()

novel_death=novel_death.drop(['Province/State','Lat','Long'],axis=1)

novel_death.columns

novel_death=novel_death.rename(columns={'Country/Region':'Region'})

#List of countries to analyse and comparing India with Rest of World for Deaths Case



list_country=['Korea, South','Italy','Spain','India','Iran']

novel_dead=novel_death[novel_death['Region'].isin(list_country)]

novel_dead_tr=novel_dead.T

novel_dead_tr

novel_dead_tr=novel_dead_tr.rename(columns={131:'India',133:'Iran',137:'Italy',143:'Korea, South',201:'Spain'})

novel_dead_tr

novel_dead_tr=novel_dead_tr.iloc[1:]#removing 1st row

novel_dead_tr

novel_dead_tr.index.name='Date'

novel_dead_tr.head()

novel_dead_tr.plot()

#Preparing Data for forecasting

df_confirmed=novel_conf_tr.copy()

df_confirmed.columns

Xdate=df_confirmed.index

Xdate

df_confirmed['DATE']=df_confirmed.index

df_confirmed.head()

date_df=df_confirmed[['DATE']]

date_df=date_df.set_index('DATE')

date_df

series_india=df_confirmed['India']

series_iran=df_confirmed['Iran']

series_spain=df_confirmed['Spain']

series_italy=df_confirmed['Italy']

series_korea=df_confirmed['Korea, South']



df_india=date_df[series_india>50]

df_india.head()

df_india['Date']=df_india.index

df_india

n=50

s_india=series_india[series_india>n]

s_india=pd.to_numeric(s_india)

s_india

india_date_s  = df_india['Date']

india_date_s

s_iran=series_iran[series_iran>n]

s_iran=pd.to_numeric(s_iran)

s_italy=series_india[series_italy>n]

s_italy=pd.to_numeric(s_india)

s_korea=series_india[series_korea>n]

s_korea=pd.to_numeric(s_korea)

Y=s_india

X = np.arange(1,len(Y)+1)

Xdate = india_date_s

Xdate

#Fitting Polynomial with degree 3

Z=np.polyfit(X,Y,3)

#Predicting the Data

P=np.poly1d(Z)

XP=np.arange(1,len(Y)+8)

YP=P(XP)#Generate Forecast

YP

Yfit=P(X)#Fit Curve

Yfit

import datetime

start = Xdate[0]

len(Xdate)

#start

end_dt = datetime.datetime.strptime(Xdate[len(Xdate)-1], "%m/%d/%y")



#end_date = datetime.datetime.strptime(str(end_dt),'%Y-%m-%d %H:%M:%S').date()



end_forecast_dt= end_dt + datetime.timedelta(days=7)



end_forecast =  datetime.datetime.strptime(str(end_forecast_dt),'%Y-%m-%d %H:%M:%S').date()

end_forecast

#

mydates = pd.date_range(start, end_forecast).to_list()

mydates_df = pd.DataFrame(mydates,columns =['Date']) 

mydates_df  = mydates_df.set_index('Date')

mydates_df['Date'] = mydates_df.index

X_FC = mydates_df['Date']

fig = plt.figure(figsize=(20,10))

ax = plt.subplot(111)

ax.plot(X, Y, '--',label='Actual Confirmed')

ax.plot(XP, YP, 'o',label='Predicted Fit using 3rd degree polynomial')

plt.title('COVID RISE IN India Current Vs Predictions till 24th May 2020')

ax.legend()

ax.set_ylim(0,300000)

ax.grid(True)

plt.show()



fig, ax = plt.subplots(figsize=(20,10))

ax.plot(X_FC,YP,'--')

ax.tick_params(direction='out', length=10, width=10, colors='r')

ax.set_xlabel('Date',fontsize=25)

ax.set_ylabel('Predicted Cases',fontsize=25)

ax.set_ylim(0,300000)

ax.set_title('COVID 19 PREDICTION for India till 09th June 2020',fontsize=25)

fig.autofmt_xdate()



ax.grid(True)

fig.tight_layout()

#Convert predicted data to datafram

dict1 = {'Date':X_FC,'Predicted_Cases':YP}

pred_df = pd.DataFrame.from_dict(dict1)

pred_df = pred_df[['Predicted_Cases']]

pred_df.Predicted_Cases = pred_df.Predicted_Cases.astype(int)

pred_df.tail(7)



from sklearn.metrics import mean_squared_error

pred_df1=pred_df.reset_index()

pred_df1=pred_df1[pred_df1['Date']<='2020-06-02']

pred_df1=pred_df1['Predicted_Cases']

pred_df1

mse=mean_squared_error(pred_df1,Y)

np.sqrt(mse)



#india_spread=df_india.groupby[]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

os.listdir('../input')

# Any results you write to the current directory are saved as output.