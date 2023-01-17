import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode
from datetime import datetime,timedelta
init_notebook_mode(connected=False)
from fbprophet import Prophet
from sklearn import metrics
import os
import lightgbm as lgb
import seaborn as sns
import plotly.express as px
import plotly.offline as py

import plotly.express as px 
import plotly.offline as py
from sklearn.preprocessing import LabelEncoder
print(os.listdir("../input/"))
path = '../input/'
# load the open line list
df_one= pd.read_csv(path+'COVID19_open_line_list.csv')
# loading confirmed cases, recovered cases and death cases from the storage
df_two = pd.read_csv(path + 'time_series_covid_19_confirmed.csv')
df_three= pd.read_csv(path+'time_series_covid_19_recovered.csv')
df_four= pd.read_csv(path + 'time_series_covid_19_deaths.csv')
# Renaming the column names to our convenience
df_two.rename({'Province/State':'State','Country/Region':'Country'},axis=1,inplace=True)
df_three.rename({'Province/State':'State','Country/Region':'Country'},axis=1,inplace=True)
df_four.rename({'Province/State':'State','Country/Region':'Country'},axis=1,inplace=True)
# Transforming the location data for better understandablility
df_two = pd.melt(df_two,id_vars=['Country','State','Lat','Long'],var_name=['Date'],value_name='Confirmed')
df_three = pd.melt(df_three,id_vars=['Country','State','Lat','Long'],var_name=['Date'],value_name='Recovered')
df_four = pd.melt(df_four,id_vars=['Country','State','Lat','Long'],var_name=['Date'],value_name='Deaths')
# Applying Joins to combine the data and making one complete data set to deal with it easily
half_set = df_two.join(df_three,lsuffix='_caller',rsuffix='_other')
data = half_set.join(df_four,lsuffix='_caller',rsuffix='_other')
# Dropping the un-necessary columns for our analysis
data = data.drop(['Country_caller', 'State_caller', 'Lat_caller', 'Long_caller',
       'Date_caller', 'Country_other', 'State_other', 'Lat_other',
       'Long_other', 'Date_other'],axis=1)
# Containing the data which we deal with, to make any decisions from now on
data = data[['Country', 'State', 'Lat',
       'Long', 'Date','Confirmed','Recovered','Deaths']]
########################33 Analysing the data of Confirmed Cases ##################################

# Dropping the columns from the Confirmed cases.
df_one = df_one.drop(['geo_resolution', 'reported_market_exposure',
       'additional_information', 'chronic_disease_binary', 'chronic_disease',
       'source', 'sequence_available', 'outcome', 'date_death_or_discharge',
       'notes_for_discussion', 'location', 'admin3', 'admin2', 'admin1',
       'country_new', 'admin_id', 'data_moderator_initials', 'Unnamed: 33',
       'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37',
       'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41',
       'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44'],axis =1)
# Renaming/assiging the confirmed cases to a better name to take off
df = df_one
# Exploratory Data Analysis

# In a sample population of the data available,
# Which placed did most of the people return from ?

df['travel_history_location'].value_counts()[:15].plot(figsize=(10, 10),kind='pie')

# Which symptoms did people show before virus hit?

df['symptoms'].value_counts()[:15].plot(figsize=(10, 10),kind='pie')
# Exploring the data based on history dates

# When did people notice the symptoms and when did they get admitted?
df_symptom_admission = df[df['date_onset_symptoms'].
                          notnull()&df['date_admission_hospital'].
                          notnull()][['date_onset_symptoms','date_admission_hospital']]

# When did they get admitted and when did they get the confirmation?
df_admission_confirm = df[df['date_confirmation'].
                          notnull()&df['date_admission_hospital'].
                          notnull()][['date_admission_hospital','date_confirmation']]

# When did they travel and when did they get the conformation?
df_travel_confirm = df[df['date_confirmation'].
                       notnull()&df['travel_history_dates'].
                       notnull()][['travel_history_dates','date_confirmation']]

# cleaning the data, making use of it.

# removing data with value none
df_symptom_admission = df_symptom_admission[df_symptom_admission['date_onset_symptoms']!='none']

# replacing bad data with the admission date
df_symptom_admission['date_admission_hospital'] = df_symptom_admission['date_admission_hospital'].replace('18.01.2020 - 23.01.2020', '18.01.2020')
df_symptom_admission['date_onset_symptoms'] = df_symptom_admission['date_onset_symptoms'].replace('10.01.2020 - 22.01.2020', '10.01.2020')
df_symptom_admission['date_onset_symptoms'] = df_symptom_admission['date_onset_symptoms'].replace('end of December 2019', '31.12.2019')


# converting date format into pandas DF format
df_symptom_admission['date_admission_hospital'] = pd.to_datetime(df_symptom_admission['date_admission_hospital'])
df_symptom_admission['date_onset_symptoms'] = pd.to_datetime(df_symptom_admission['date_onset_symptoms'])


# calculating the time difference of the two comulns,converting into int from float
df_symptom_admission['diff'] = ((df_symptom_admission['date_admission_hospital']-df_symptom_admission['date_onset_symptoms']).dt.total_seconds()/(3600*24)).apply(np.floor).astype(int)

# considering only meaningful data, because there exist quite a bit of incorrect entries
df_symptom_admission = df_symptom_admission[df_symptom_admission['diff']>0]

# Repeting the above process for the rest of the two data frames for the analysis
# Admission and Conformation data
df_admission_confirm['date_admission_hospital'] = df_admission_confirm['date_admission_hospital'].replace('18.01.2020 - 23.01.2020', '18.01.2020')
df_admission_confirm['date_admission_hospital'] = pd.to_datetime(df_admission_confirm['date_admission_hospital'])
df_admission_confirm['date_confirmation'] = pd.to_datetime(df_admission_confirm['date_confirmation'])
df_admission_confirm['diff'] = ((df_admission_confirm['date_confirmation']-df_admission_confirm['date_admission_hospital']).
                                    dt.total_seconds()/(3600*24)).apply(np.floor).astype(int)
df_admission_confirm = df_admission_confirm[df_admission_confirm['diff']>0]

# Travel & Confirmation data
df_travel_confirm = df_travel_confirm[df_travel_confirm['travel_history_dates'].str.len()==10]
df_travel_confirm['travel_history_dates'] = pd.to_datetime(df_travel_confirm['travel_history_dates'].str.replace(',','.'))
df_travel_confirm['date_confirmation'] = pd.to_datetime(df_travel_confirm['date_confirmation'])
df_travel_confirm['diff'] = ((df_travel_confirm['date_confirmation']-df_travel_confirm['travel_history_dates']).
                             dt.total_seconds()/(3600*24)).apply(np.floor).astype(int)
df_travel_confirm = df_travel_confirm[df_travel_confirm['diff']>0]

# What is the time duration between symptoms discovered and the person's admission into the hospital ?

ax = df_symptom_admission['diff'].value_counts()[:15].plot(figsize=(15, 7),kind='bar',title='Duration between symptoms till admission')
ax.set_xlabel('Days from symptoms till Hospital Admit')
ax.set_ylabel('Number of cases within this delay')
# How long did it take to confirm if the patient has been detected with the virus?

ax = df_admission_confirm['diff'].value_counts()[:15].plot(figsize=(15, 7),kind='bar',title='Duration between admission till conformation')
ax.set_xlabel('Days from admission till confirmation')
ax.set_ylabel('Number of cases within these days')
# With in how many days of travel return, person got detected with the virus ?

ax = df_travel_confirm['diff'].value_counts()[:15].plot(figsize=(15, 7),kind='bar',title='Duration between travel return till conformation')
ax.set_xlabel('Days from travel return till confirmation')
ax.set_ylabel('Number of cases within these days')
# What is the Male/ Female ratio of the virus detection?

df.sex.fillna('Unknown',inplace=True)
df.sex=df['sex'].map({"Female":"female","Male":"male","male":"male",'female':'female',"Unknown":"Unknown"})

sex= df.sex.value_counts()[1:]
fig=px.pie(sex,sex.index,sex)
fig.update_layout(title="Male vs Female infected Globally")
# melt_and_merge() function is responsible for transforming the data and make it more understandable and meaningful for the anaysis
def melt_and_merge(agg=True):
    
    # 4 step proces --> Reading data, Transforming it, Renaming, convert the date format into pandas datetime format
    df_one = pd.read_csv(path+'time_series_covid_19_recovered.csv')
    df_one=pd.melt(df_one,id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name=['Date'],value_name='Recovered')
    df_one.rename({'Province/State':"State","Country/Region":'Country'},axis=1,inplace=True)
    df_one['Date']=pd.to_datetime(df_one['Date'])

    df_two=pd.read_csv(path+"time_series_covid_19_deaths.csv")
    df_two=pd.melt(df_two,id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name=['Date'],value_name='Deaths')
    df_two.rename({'Province/State':"State","Country/Region":'Country'},axis=1,inplace=True)
    df_two['Date']=pd.to_datetime(df_two['Date'])

    df_three=pd.read_csv(path+"time_series_covid_19_confirmed.csv")
    df_three=pd.melt(df_three,id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name=['Date'],value_name='Confirmed')
    df_three.rename({'Province/State':"State","Country/Region":'Country'},axis=1,inplace=True)
    df_three['Date']=pd.to_datetime(df_three['Date'])
    
    if (agg):
        col={"Lat":np.mean,"Long":np.mean,"Recovered":sum}
        df_one=df_one.groupby(['Country',"Date"],as_index=False).agg(col)
        
        col={"Lat":np.mean,"Long":np.mean,"Deaths":sum}
        df_two=df_two.groupby(['Country',"Date"],as_index=False).agg(col)
        
        col={"Lat":np.mean,"Long":np.mean,"Confirmed":sum}
        df_three=df_three.groupby(['Country',"Date"],as_index=False).agg(col)

    else:
        df_one['State'].fillna(df_one['Country'],inplace=True)
        df_two['State'].fillna(df_two['Country'],inplace=True)
        df_three['State'].fillna(df_three['Country'],inplace=True)
    
    
    print("The shape of three datasets are equal :",(df_three.shape[0]==df_one.shape[0]==df_two.shape[0]))
    
    merge=pd.merge(df_one,df_two)
    merge=pd.merge(merge,df_three)
    
    return merge
# Asking to melt_and_merge aggregate data
data=melt_and_merge(True)
# How many people got effected with Deaths in the whole world(Just top 15)?

x=data.groupby(['Country'],as_index=False)['Deaths'].last().sort_values(by="Deaths",ascending=False)[:15]
fig=px.pie(x,"Country","Deaths")
fig.update_layout(title="Global Covid-19 Deaths")
# Statistics from different parts of the world 

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# we are expecting 3 Bar plots in a single column in a single plot to compare and interpret
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Confirmed", "Recovered","Deaths"))

effect_type = ["Confirmed", "Recovered","Deaths"]

confirmed_effected=data.groupby(['Country'],as_index=False)['Confirmed'].last().sort_values(by="Confirmed",ascending=False)[:10]
fig.add_trace(go.Bar(x=confirmed_effected['Country'], y=confirmed_effected['Confirmed'],
                    marker=dict(color=confirmed_effected['Confirmed'], coloraxis="coloraxis"))
              , row=1, col=1)
recover_effected=data.groupby(['Country'],as_index=False)['Recovered'].last().sort_values(by="Recovered",ascending=False)[:10]
fig.add_trace(go.Bar(x=recover_effected['Country'], y=recover_effected['Recovered'],
                    marker=dict(color=recover_effected['Recovered'], coloraxis="coloraxis")),
               row=2, col=1)

dead_effected=data.groupby(['Country'],as_index=False)['Deaths'].last().sort_values(by="Deaths",ascending=False)[:10]
fig.add_trace(go.Bar(x=dead_effected['Country'], y=dead_effected['Deaths'],
                    marker=dict(color=dead_effected['Deaths'], coloraxis="coloraxis")),
               row=3, col=1)

fig.update_layout(height=900, width=1000, title_text="Stacked subplots")
fig.show() 
# We see that major stake of recovery is taken over by china, 
# so let us exclude china for a moment from our visualizations and compare the other countries performances

dt_noChina = data[data['Country']!='China']

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Confirmed", "Recovered","Deaths"))

effect_type = ["Confirmed", "Recovered","Deaths"]

confirmed_effected=dt_noChina.groupby(['Country'],as_index=False)['Confirmed'].last().sort_values(by="Confirmed",ascending=False)[:10]
fig.add_trace(go.Bar(x=confirmed_effected['Country'], y=confirmed_effected['Confirmed'],
                    marker=dict(color=confirmed_effected['Confirmed'], coloraxis="coloraxis"))
              , row=1, col=1)
recover_effected=dt_noChina.groupby(['Country'],as_index=False)['Recovered'].last().sort_values(by="Recovered",ascending=False)[:10]
fig.add_trace(go.Bar(x=recover_effected['Country'], y=recover_effected['Recovered'],
                    marker=dict(color=recover_effected['Recovered'], coloraxis="coloraxis")),
               row=2, col=1)

dead_effected=dt_noChina.groupby(['Country'],as_index=False)['Deaths'].last().sort_values(by="Deaths",ascending=False)[:10]
fig.add_trace(go.Bar(x=dead_effected['Country'], y=dead_effected['Deaths'],
                    marker=dict(color=dead_effected['Deaths'], coloraxis="coloraxis")),
               row=3, col=1)

fig.update_layout(height=900, width=1000, title_text="Stacked subplots")
fig.show()
# Let us observe the top 5 countries statistics in all the categories
top5_confirmed = data.groupby(['Country'],as_index=False)['Confirmed'].last().sort_values(by="Confirmed",ascending=False)[:5]['Country'].to_numpy()
top5_recovered = data.groupby(['Country'],as_index=False)['Recovered'].last().sort_values(by="Recovered",ascending=False)[:5]['Country'].to_numpy()
top5_dead = data.groupby(['Country'],as_index=False)['Deaths'].last().sort_values(by="Deaths",ascending=False)[:5]['Country'].to_numpy()
# Confirmed statistics from top 5 countries

fig = go.Figure()
for country in range(len(top5_confirmed)):
    fig.add_trace(go.Scatter(
        x=data[data['Country']==top5_confirmed[country]]['Date'],
        y=data[data['Country']==top5_confirmed[country]]['Confirmed'],
        name = top5_confirmed[country],
        connectgaps=True 
    ))
fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)  
fig.update_layout(height=500, width=1000)  
fig.show()
# Recovered Statistics from top 5 countries

fig = go.Figure()
for country in range(len(top5_recovered)):
    fig.add_trace(go.Scatter(
        x=data[data['Country']==top5_recovered[country]]['Date'],
        y=data[data['Country']==top5_recovered[country]]['Recovered'],
        name = top5_recovered[country],
        connectgaps=True 
    ))
fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)  
fig.update_layout(height=500, width=1000)  
fig.show()
# Death statistics from top 5 countries

fig = go.Figure()
for country in range(len(top5_dead)):
    fig.add_trace(go.Scatter(
        x=data[data['Country']==top5_dead[country]]['Date'],
        y=data[data['Country']==top5_dead[country]]['Deaths'],
        name = top5_dead[country],
        connectgaps=True 
    ))  
fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)  
fig.update_layout(height=500, width=1000)  
fig.show()

# Since China seem to be the majorly impacted one lets deep dive into interesting facts of China
china=data[data['Country']=="China"]
fig = go.Figure()
for i in ["Confirmed","Recovered","Deaths"]:
    fig.add_trace(go.Scatter(
        y=china[i],
        x=china['Date'],
        name = i, # Style name/legend entry with html tags
        connectgaps=True # override default to connect the gaps
    ))
fig.update_layout(title="Timeseries plot of China ") 
fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)
fig.show()
# There are multiple states inside china so run the function with False parameter, we do not want the aggregated data here
df=melt_and_merge(False)
# Because city Hubei might have very high numbers, may it is quite not interesting to view the graph
china=df[df['Country']=="China"]
fig = go.Figure()
states=china.State.unique().tolist()
for state in states:
    fig.add_trace(go.Scatter(
        x=china[china['State']==state]['Date'],
        y=china[china['State']==state]['Confirmed'],
        name = state, # Style name/legend entry with html tags
        connectgaps=True # override default to connect the gaps
    ))
fig.update_layout(title="Timeseries plot of number of Confirmed Cases in all the Provinces")  
fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')
fig.update_traces(mode='lines', marker_line_width=2.5, marker_size=3)

fig.show()
# Let us exclude, Hubei from our analysis for now, lets check the plot of confirmed cases
china=df[df['Country']=="China"]
fig = go.Figure()
states=china.State.unique().tolist()
states.remove('Hubei')
for state in states:
    fig.add_trace(go.Scatter(
        x=china[china['State']==state]['Date'],
        y=china[china['State']==state]['Confirmed'],
        name = state, # Style name/legend entry with html tags
        connectgaps=True # override default to connect the gaps
    ))
fig.update_layout(title="Timeseries plot of number of Confirmed Cases in all the Provinces")  
fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')
fig.update_traces(mode='lines', marker_line_width=2.5, marker_size=3)

fig.show()
# we could also explore the Death and Recovered cases of the other plases like above.


# Getting only the data of china to analyse
china=df[df['Country']=="China"].groupby(['State'],as_index=False)[[ 'Lat', 'Long', 'Date', 'Recovered', 'Deaths',"Confirmed"]].last()
# Hubei stands out ..! 

fig=px.scatter(china,x="State",y="Confirmed")
fig.update_layout(title="Confirmed Cases in States of China")
fig.show()

# Why did that happen ?
# Did it happen over night in Hubei ?
# Nooo..!!

# Learnign about lag on confirmed, recovere, death cases respectively

Hubei=df[df['State']=="Hubei"]
Hubei.loc[:,'lag_1']=Hubei['Confirmed'].shift(1)
Hubei.loc[:,'NewCases']=(Hubei['Confirmed']-Hubei['lag_1']).fillna(0).values
fig=px.bar(Hubei,x="Date",y="NewCases")
fig.update_layout(title="New Confirmed Cases in Hubei province")
fig.show()
Hubei=df[df['State']=="Hubei"]
Hubei.loc[:,'lag']=Hubei['Recovered'].shift(1)
Hubei.loc[:,'CurrentDateRecovery']=(Hubei['Recovered']-Hubei['lag']).fillna(0)
fig=px.bar(Hubei,x="Date",y="CurrentDateRecovery")
fig.update_layout(title="Current day recovered cases in Hubei province")
fig.show()
Hubei=df[df['State']=="Hubei"]
Hubei.loc[:,'lag']=Hubei['Deaths'].shift(1)
Hubei.loc[:,'CurrentDateDeaths']=(Hubei['Deaths']-Hubei['lag']).fillna(0)
fig=px.bar(Hubei,x="Date",y="CurrentDateDeaths")
fig.update_layout(title="Everyday Death cases in Hubei province")
fig.show()
# But what about the rest of the world
restof_world=data[data['Country']!="China"].groupby(['Date'],as_index=False)[['Confirmed',"Recovered","Deaths"]].agg(sum)
china=data[data['Country']=="China"]

fig = go.Figure()
fig.add_trace(go.Bar(x=china['Date'],
                y=china['Confirmed'],
                name='China',
                marker_color='rgb(255, 0, 0)'
                ))
fig.add_trace(go.Bar(x=restof_world['Date'],
                y=restof_world['Confirmed'],
                name='Rest of world',
                marker_color='rgb(0, 0, 255)'
                ))

fig.update_layout(
    title='Global Confirmed Cases, China and Rest of World',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Confirmed Cases',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()
fig = go.Figure()
fig.add_trace(go.Bar(x=china['Date'],
                y=china['Deaths'],
                name='China',
                marker_color='rgb(255, 0, 0)'
                ))
fig.add_trace(go.Bar(x=restof_world['Date'],
                y=restof_world['Deaths'],
                name='Rest of world',
                marker_color='rgb(0, 0, 255)'
                ))

fig.update_layout(
    title='Global Deaths China and Rest of World',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Death Cases',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(250, 242, 242,0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()
fig = go.Figure()
fig.add_trace(go.Bar(x=china['Date'],
                y=china['Recovered'],
                name='China',
                marker_color='rgb(255, 0, 0)'
                ))
fig.add_trace(go.Bar(x=restof_world['Date'],
                y=restof_world['Recovered'],
                name='Rest of world',
                marker_color='rgb(0, 0, 255)'
                ))

fig.update_layout(
    title='Global Recovered Cases, China and Rest of World',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Recovered Cases',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

fig.show()
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Italy","Korea, South","Spain", "Germany",))

countries=["Italy","Korea, South","Spain", "Germany",]

    
country=data[data['Country']==countries[0]]
fig.add_trace(go.Scatter(x=country['Date'], y=country['Confirmed'],
                    marker=dict(color=country['Confirmed'], coloraxis="coloraxis")),
              1, 1)
    
country=data[data['Country']==countries[1]]
fig.add_trace(go.Scatter(x=country['Date'], y=country['Confirmed'],
                    marker=dict(color=country['Confirmed'], coloraxis="coloraxis")),
              1,2 )
    
country=data[data['Country']==countries[2]]
fig.add_trace(go.Scatter(x=country['Date'], y=country['Confirmed'],
                    marker=dict(color=country['Confirmed'], coloraxis="coloraxis")),
              2, 1)
    
country=data[data['Country']==countries[3]]
fig.add_trace(go.Scatter(x=country['Date'], y=country['Confirmed'],
                    marker=dict(color=country['Confirmed'], coloraxis="coloraxis")),
              2,2 )
fig.update_layout(title="Confirmed cases in Italy, South Korea, Spain and Germany")

fig.show() 
 #################   Time to make a model ###########
data=pd.read_csv(path+"covid_19_data.csv")
data.isna().sum()
data.fillna("Unknown",inplace=True)
data=data[["ObservationDate","Province/State","Country/Region","Confirmed","Deaths","Recovered"]]

data.rename({"ObservationDate":"ds","Province/State":"State","Country/Region":"Country"},axis=1,inplace=True)
data['ds']=pd.to_datetime(data['ds'])
# Separates data according to the date but not at random

def train_test_split(df,test_days):
    df=data.copy()
    max_date=df.ds.max()-timedelta(test_days)
    
    for col in ["State","Country"]:
        lb=LabelEncoder()
        df[col]=lb.fit_transform(df[col])
    
    train = df[df['ds'] < max_date]
    
    test = df[df['ds'] > max_date]
    
    return train,test
# Separate the last 7 days of data to check the accuracy of the model

train,test= train_test_split(data,7)
# Applying 1st model of Forcasting time series using Prophet Algorithm

def train_predict(train,test):
    targets=['Confirmed',"Deaths","Recovered"]
    predictions=pd.DataFrame()
    for col in targets:
        
        trainX=train[['ds',"State","Country"]+[col]]
        X_test=test[['ds','State', 'Country']]
        
        m= Prophet()
        trainX.rename({col:"y"},axis=1,inplace=True)
        m.add_regressor("State")
        m.add_regressor("Country")
        m.fit(trainX)
        
        future=m.predict(X_test)
        
        predictions[col]=future['yhat']
        
    return predictions
sub=train_predict(train,test)
sub.head()
# Feature Engineering
def simple_fe(df):
    
    df['year']=df['ds'].dt.year
    df['month']=df['ds'].dt.month
    df['day']=df['ds'].dt.day
    
    ##lag features
    df.loc[:,'rec_lag_2']=df.groupby(['Country','State'])['Recovered'].transform(lambda x: x.shift(1))
    df.loc[:,'conf_lag_2'] = df.groupby(['Country'])['Confirmed'].transform(lambda x: x.shift(1))
    df.loc[:,'deaths_lag_2'] =df.groupby(['Country'])['Deaths'].transform(lambda x: x.shift(1))
    
    ##rolling mean
    df['rec_rollmean_7']=df.groupby(['Country','State'])['Recovered'].transform(lambda x: x.rolling(7).mean())
    df['conf_rollmean_7'] = df.groupby(['Country'])['Confirmed'].transform(lambda x: x.rolling(7).mean())
    df['deaths_rollmean_7'] =df.groupby(['Country'])['Deaths'].transform(lambda x: x.rolling(7).mean())
    
    ##rolling std
    df['rec_rollstd_7']=df.groupby(['Country','State'])['Recovered'].transform(lambda x: x.rolling(7).std())
    df['conf_rollstd_7'] = df.groupby(['Country'])['Confirmed'].transform(lambda x: x.rolling(7).std())
    df['deaths_rollstd_7'] =df.groupby(['Country'])['Deaths'].transform(lambda x: x.rolling(7).std())
    
    #df.drop(['ds'],axis=1,inplace=True)
    df.fillna(0,inplace=True)
    
    return df
    
data= simple_fe(data)
# Applying LightGradientBooster Algorithm
def run_lgb(data,target):
    
    features=['year', 'month','State', 'Country','Recovered',
               'day', 'rec_lag_2', 'conf_lag_2', 'deaths_lag_2',
               'rec_rollmean_7', 'conf_rollmean_7', 'deaths_rollmean_7',
               'rec_rollstd_7', 'conf_rollstd_7', 'deaths_rollstd_7']
     
    train,test=train_test_split(data,7)
    x_train=train[features]
    y_train=train[target]
    print(x_train.shape)
    x_val=test[features]
    y_val=test[target]
    print(x_val.shape)

    # define random hyperparammeters
    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'n_jobs': -1,
        'seed': 236,
        'learning_rate': 0.1,
        'bagging_fraction': 0.75,
        'bagging_freq': 10, 
        'colsample_bytree': 0.75}

    train_set = lgb.Dataset(x_train[features], y_train,categorical_feature=['State',"Country",'year','month','day'])
    val_set = lgb.Dataset(x_val[features], y_val,categorical_feature=['State',"Country",'year','month','day'])

    del x_train, y_train

    model = lgb.train(params, train_set, num_boost_round = 500, early_stopping_rounds = 50, valid_sets = [train_set, val_set],
                      verbose_eval = 100,)
    val_pred = model.predict(x_val[features])
    val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
    print(val_score)

    #y_pred = model.predict(x_val)
    #test[targets] = y_pred.values
    return val_pred
sub=pd.DataFrame()
sub['ds']=test['ds'].values
targets=['Confirmed', 'Deaths', 'Recovered']
for target in targets:
    
        sub[target]=run_lgb(data,target)
sub.head()