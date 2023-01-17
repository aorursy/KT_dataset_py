import os
from requests import request
import urllib.request
import json
from pandas.io.json import json_normalize

import numpy as np
import pandas as pd
import pandas_profiling
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import matplotlib.pyplot as plt
import folium 
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# color pallette
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 

# hide warnings
import warnings
warnings.filterwarnings('ignore')

from IPython.display import Markdown

%matplotlib inline
register_matplotlib_converters()
def bold(string):
    display(Markdown(string))
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
def read_from_api(URL, x=None):
    """
    Read data from API and Return Normalized JSON
    
    Keyword arguments:
    URL -- String API URL
    x -- String name to normalize API request into JSON
    """
    response = request(url=URL, method='get')
    elevations = response.json()
    return json_normalize(elevations) if x==None else json_normalize(elevations[x])
''' Function to plot countplot between to columns with bins valaues [0,20,30,40,50,60,70,80,90,100] '''

def countplot(columnname1,columnname2,plotTitle):
    bins = [0,20,30,40,50,60,70,80,90,100]
    plt.figure(figsize = (14,8))
    sns.countplot(x=pd.cut(columnname1,bins), hue = columnname2 , orient = 'h')
    plt.xlabel(columnname1.name)
    plt.yscale('log')
    plt.title(plotTitle)
    plt.grid(True)
    plt.show()
    return
''' Function to plot boxplot between two columns '''

def boxplot(dfname,columnname1,columnname2,plotTitle):
  plt.figure(figsize=(12, 6), dpi = 100)
  sns.boxplot(x = columnname1, y = columnname2, data = dfname, palette = 'viridis')
  plt.title(plotTitle)
  plt.xlabel(columnname1)
  plt.xticks(rotation=90) 
  plt.ylabel(columnname2)
  plt.tight_layout()
  plt.show()
  return
''' Function to plot pieChart '''
  
def pieChart(dfname,columnname, plotTitle):
    fig = px.pie(dfname, values=columnname, names=dfname.index
             ,color_discrete_sequence=px.colors.sequential.Plasma_r,title=plotTitle)
    fig.update_traces(textposition='outside', textinfo='value+label')
    fig.show()
    return
''' Function to plot bar chart'''

def barChart(dfname, columnname1, columnname2, plotTitle, barOrientation, color=None, width=800, height=800):
  fig = px.bar(dfname, x=columnname1, y=columnname2, color=color, orientation=barOrientation, text=columnname1, width=width,height=height,
       color_discrete_sequence = ['#35495e'], title=plotTitle)
  fig.update_xaxes(title=columnname1)
  fig.update_yaxes(title=columnname2)
  fig.show()
  return
''' Function to plot Histogram Distribution'''

def histogramChart(dfname , columnname , plotTitle):
    fig = px.histogram(dfname, x=columnname, color_discrete_sequence = ['#35495e'], nbins=50,title=plotTitle)
    fig.show()
    return
''' Function to plot Tree Map'''

def treeMapCart(dfname , columnList , valueColumn , plotTitle):
  fig = px.treemap(dfname, path=columnList, values=valueColumn, height=700,
           title=plotTitle, color_discrete_sequence = px.colors.qualitative.Prism)
  fig.data[0].textinfo = 'label+text+value'
  fig.show()
  return
df_raw_data1 = read_from_api('https://api.covid19india.org/raw_data1.json', 'raw_data')
df_raw_data2 = read_from_api('https://api.covid19india.org/raw_data2.json', 'raw_data')
df_raw_data3 = read_from_api('https://api.covid19india.org/raw_data3.json', 'raw_data')
df_raw_data4 = read_from_api('https://api.covid19india.org/raw_data4.json', 'raw_data')
df_raw_data5 = read_from_api('https://api.covid19india.org/raw_data5.json', 'raw_data')
df_raw_data6 = read_from_api('https://api.covid19india.org/raw_data6.json', 'raw_data')
df_raw_data7 = read_from_api('https://api.covid19india.org/raw_data7.json', 'raw_data')
sum = df_raw_data1.shape[0] + df_raw_data2.shape[0] + df_raw_data3.shape[0] + df_raw_data5.shape[0] \
    + df_raw_data6.shape[0] +df_raw_data7.shape[0] +df_raw_data4.shape[0]
print(sum)
# Merge Patient Level Raw Data 
df_raw_data = df_raw_data1.append(df_raw_data2)
df_raw_data = df_raw_data.append(df_raw_data3)
df_raw_data = df_raw_data.append(df_raw_data4)
df_raw_data = df_raw_data.append(df_raw_data5)
df_raw_data = df_raw_data.append(df_raw_data6)
df_raw_data = df_raw_data.append(df_raw_data7)

df_raw_data.shape
df_death_and_recoveries = read_from_api('https://api.covid19india.org/deaths_recoveries.json','deaths_recoveries')
df_cases_time_series = read_from_api('https://api.covid19india.org/data.json','cases_time_series')
df_statewise = read_from_api('https://api.covid19india.org/data.json','statewise')
df_tested = read_from_api('https://api.covid19india.org/data.json','tested')
df_district_wise = read_from_api(URL='https://api.covid19india.org/v2/state_district_wise.json')
df_states_daily = read_from_api('https://api.covid19india.org/states_daily.json','states_daily')
df_resources = read_from_api('https://api.covid19india.org/resources/resources.json','resources')
bold('**COVID19 - RAW DATA**')
df_raw_data.head()
bold('**COVID19 - DEATH AND RECOVERY DATA**')
df_death_and_recoveries.head()
bold('**COVID19 - CASES TIME SERIES DATA**')
df_cases_time_series.head()
bold('**COVID19 - STATEWISE DATA**')
df_statewise.head()
bold('**COVID19 - TESTS DATA**')
df_tested.head()
bold('**COVID19 - DISTRICTWISE DATA**')
df_district_wise.head()
bold('**COVID19 - STATES DATA**')
df_states_daily.head()
bold('**COVID19 - ESSENTIALS AND RESOURCES DATA**')
df_resources.head()
bold("**Data Shape : Rows = {} , Columns = {}**".format(df_raw_data.shape[0],df_raw_data.shape[1]))
print("Column Names are : \n", df_raw_data.columns)
data=df_raw_data.copy()
df_raw_data.describe()
df_raw_data.info()
df_raw_data['agebracket'] = df_raw_data['agebracket'].replace('28-35', 35)
# df_raw_data['agebracket'] = df_raw_data['agebracket'].astype(int)
df_raw_data['agebracket'] = pd.to_numeric(df_raw_data['agebracket'], errors='coerce')
df_raw_data['backupnotes'] = df_raw_data['backupnotes'].astype(str)
df_raw_data['contractedfromwhichpatientsuspected'] = df_raw_data['contractedfromwhichpatientsuspected'].astype(str)
df_raw_data['currentstatus'] = df_raw_data['currentstatus'].astype('category')
df_raw_data['dateannounced'] = pd.to_datetime(df_raw_data['dateannounced'])
df_raw_data['detectedcity'] = df_raw_data['detectedcity'].astype(str)
df_raw_data['detecteddistrict'] = df_raw_data['detecteddistrict'].astype(str)
df_raw_data['detectedstate'] = df_raw_data['detectedstate'].astype(str)
df_raw_data['gender']= df_raw_data['gender'].astype('category')
df_raw_data['nationality']=df_raw_data['nationality'].astype(str)
df_raw_data['notes']= df_raw_data['notes'].astype('category')
df_raw_data['patientnumber'] = pd.to_numeric(df_raw_data['patientnumber'],errors='coerce')
df_raw_data['source1']=df_raw_data['source1'].astype(str)
df_raw_data['source2']=df_raw_data['source2'].astype(str)
df_raw_data['source3']=df_raw_data['source3'].astype(str)
df_raw_data['statecode']=df_raw_data['statecode'].astype(str)
df_raw_data['statepatientnumber']=df_raw_data['statepatientnumber'].astype(str)
df_raw_data['statuschangedate']=pd.to_datetime(df_raw_data['statuschangedate'])
df_raw_data['typeoftransmission']=df_raw_data['typeoftransmission'].astype('category')
df_raw_data.drop(['estimatedonsetdate', 'contractedfromwhichpatientsuspected', 'source1', 'source2', 'source3', 'backupnotes' ], axis = 1, inplace = True)
df_raw_data.sample(10)
missing_data(df_raw_data)
# profile = pandas_profiling.ProfileReport(df_raw_data)
# profile.to_file(outputfile="covid19_data_after_preprocessing.html")
# pandas_profiling.ProfileReport(df_raw_data)
plt.figure(figsize=(12,6), dpi = 100)
countplot(df_raw_data["agebracket"],df_raw_data["currentstatus"],"Age range with Covid-19 patient")
plt.figure(figsize=(12,6), dpi = 100)
boxplot(df_raw_data,"nationality","agebracket","Covid19 - Age Range distribution across Nationality")
state = df_raw_data.groupby('detectedstate').count()
pieChart(state , 'currentstatus' ,'Covid19 cases based on State')
temp = df_raw_data.groupby('nationality')['patientnumber'].count().reset_index()
temp = temp.sort_values('patientnumber')
temp = temp[temp['nationality']!='']
temp = temp[temp['nationality']!='India']

barChart(temp , 'patientnumber' , 'nationality' , 'No. of foreign citizens' ,'h', color='patientnumber' )
temp = pd.DataFrame(df_raw_data[['typeoftransmission']].groupby('typeoftransmission')['typeoftransmission'].count())
print(temp)
temp = temp.dropna()
temp.columns = ['count']
temp = temp.reset_index().sort_values(by='count')

barChart(temp , 'count' , 'typeoftransmission' , 'Type of transmission','h', color = 'count' )
fig = plotly.subplots.make_subplots(
    rows=1, cols=2, column_widths=[0.8, 0.2],
    subplot_titles = ['Cases vs Age', ''],
    specs=[[{"type": "histogram"}, {"type": "pie"}]]
)

temp = df_raw_data[['agebracket', 'currentstatus']].dropna()
print('Total no. of values :', df_raw_data.shape[0], '\nNo. of missing values :', df_raw_data.shape[0]-temp.shape[0], '\nNo. of available values :', df_raw_data.shape[0]-(df_raw_data.shape[0]-temp.shape[0]))
gen_grp = temp.groupby('currentstatus').count()

fig.add_trace(go.Pie(values=gen_grp.values.reshape(-1).tolist(), labels=['Deceased', 'Hospitalized', 'Recovered'], 
                     marker_colors = ['#fd0054', '#393e46', '#40a798'], hole=.3),1, 2)

fig.add_trace(go.Histogram(x=temp[temp['currentstatus']=='Deceased']['agebracket'], nbinsx=50, name='Deceased', marker_color='#808080'), 1, 1)
fig.add_trace(go.Histogram(x=temp[temp['currentstatus']=='Recovered']['agebracket'], nbinsx=50, name='Recovered', marker_color='#008000'), 1, 1)
fig.add_trace(go.Histogram(x=temp[temp['currentstatus']=='Hospitalized']['agebracket'], nbinsx=50, name='Hospitalized', marker_color='#FF0000'), 1, 1)

fig.update_layout(showlegend=False)
fig.update_layout(barmode='stack')
fig.data[0].textinfo = 'label+text+value+percent'

fig.show()
fig = plotly.subplots.make_subplots(
    rows=1, cols=2, column_widths=[0.8, 0.2],
    subplot_titles = ['Gender vs Age', ''],
    specs=[[{"type": "histogram"}, {"type": "pie"}]]
)

temp = df_raw_data[['agebracket', 'gender']].dropna()
print('Total no. of values :', df_raw_data.shape[0], '\nNo. of missing values :', df_raw_data.shape[0]-temp.shape[0], '\nNo. of available values :', df_raw_data.shape[0]-(df_raw_data.shape[0]-temp.shape[0]))
gen_grp = temp.groupby('gender').count()

fig.add_trace(go.Histogram(x=temp[temp['gender']=='F']['agebracket'], nbinsx=50, name='Female', marker_color='#008000'), 1, 1)
fig.add_trace(go.Histogram(x=temp[temp['gender']=='M']['agebracket'], nbinsx=50, name='Male', marker_color='#FF0000'), 1, 1)

fig.add_trace(go.Pie(values=gen_grp.values.reshape(-1).tolist(), labels=['Female', 'Male'], marker_colors = ['#008000', '#FF0000']),1, 2)

fig.update_layout(showlegend=False)
fig.update_layout(barmode='stack')
fig.data[2].textinfo = 'label+text+value+percent'

fig.show()
print('Total no. of values :', df_raw_data.shape[0], '\nNo. of missing values :', df_raw_data.shape[0]-df_raw_data[['agebracket']].dropna().shape[0],
      '\nNo. of available values :', df_raw_data.shape[0]-(df_raw_data.shape[0]-df_raw_data[['agebracket']].dropna().shape[0]))

histogramChart(df_raw_data,'agebracket','Distribution of ages of confirmed patients',)
dist = df_raw_data.groupby(['detectedstate', 'detecteddistrict'])['patientnumber'].count().reset_index()
dist.head()

treeMapCart(dist, ['detectedstate', 'detecteddistrict'] , 'patientnumber' , 'Number of Confirmed Cases')
plt.figure(figsize=(12,6), dpi = 100)
boxplot(df_raw_data,'detectedstate' , 'agebracket' ,'Age Distribution of Detected Cases acros States and UT')
plt.figure(figsize=(12, 6), dpi = 100)
boxplot(df_raw_data , 'nationality' , 'agebracket' , "Age Distribution across different Nationality")
dist = df_raw_data.groupby(['agebracket','currentstatus'])['patientnumber'].count().reset_index()
dist = dist[dist['currentstatus']=='Recovered']
dist
fig = px.bar(dist, x='agebracket', y='patientnumber', orientation='v', text='patientnumber', width=1200,
       color_discrete_sequence = ['#00CC96'], title='Age distribution of Recovered COVID Patient')

fig.update_xaxes(title='Age')
fig.update_yaxes(title='# Patient')
fig.show()

dist = df_raw_data.groupby(['gender','currentstatus'])['patientnumber'].count().reset_index()
dist = dist[dist['currentstatus']=='Recovered']
dist = dist[dist['gender'] != ""]
print(dist)


fig = px.pie(dist, values=dist['patientnumber'], names=dist.gender
         ,color_discrete_sequence=["#636EFA"],title='Gender distribution of COVID19 Recovered Patients')
fig.update_traces(textposition='outside', textinfo='value+label')
fig.show()
detected_city = df_raw_data['detectedcity'].value_counts().reset_index()
detected_city.rename(columns={"index":"cities",
                            "detectedcity": "Counts"}, inplace=True)

detected_city.cities.replace('', np.nan, inplace=True)
detected_city.dropna(subset=['cities'], how='all', inplace=True)
detected_city.head()

#barChart(detected_city, 'cities', 'Counts', 'Hotspot Cities Detected With Most cases', 'v')

fig = px.bar(detected_city.sort_values('Counts', ascending=True).sort_values('Counts', ascending=False).head(15), 
             y="Counts", x="cities", color= "cities",
             title='Hotspot Cities Detected With Most cases', 
             orientation='v',
             color_discrete_sequence = px.colors.cyclical.IceFire,
             width=700, height=600)
fig.update_traces(opacity=0.8)
fig.update_xaxes(title='Cities')
fig.update_yaxes(title='Counts')
fig.update_layout(template = 'plotly_white')
fig.show()
df_statewise.info()
df_statewise.head()
print("Data Shape : Rows = {} , Columns = {}".format(df_statewise.shape[0],df_statewise.shape[1]))
print("Column Names are : \n", df_statewise.columns)
cols = ['active', 'confirmed', 'deaths', 'deltaconfirmed', 'deltadeaths',
       'deltarecovered', 'recovered']
df_statewise['lastupdatedtime'] = pd.to_datetime(df_statewise['lastupdatedtime'])
df_statewise[cols] = df_statewise[cols].astype(int)
df_statewise.info()
statewise_cases = df_statewise[['state','active','confirmed','deaths','recovered']]
statewise_cases = statewise_cases[statewise_cases.state !='Total']
statewise_cases['death_rate (per 100)'] = np.round(100*statewise_cases['deaths']/statewise_cases['confirmed'],2)
statewise_cases['recovery_rate (per 100)'] = np.round(100*statewise_cases['recovered']/statewise_cases['confirmed'],2)
statewise_cases.head()
statewise_cases.dropna(subset=['death_rate (per 100)','recovery_rate (per 100)'], how='all', inplace=True)
print('Total Confirmed Cases: ',statewise_cases['confirmed'].sum())
print('Total Deaths: ',statewise_cases['deaths'].sum())
print('Total Recovered Cases: ',statewise_cases['recovered'].count())
print('Death Rate (per 100): ',np.round(100*statewise_cases['deaths'].sum()/statewise_cases['confirmed'].sum(),2))
print('Recovery Rate (per 100): ',np.round(100*statewise_cases['recovered'].sum()/statewise_cases['confirmed'].sum(),2))
bold("**STATE WISE CONFIRMED, DEATH AND RECOVERED CASES OF COVID-19**")
statewise_cases.sort_values('confirmed', ascending= False).style.background_gradient(cmap='YlOrBr',subset=["confirmed"])\
                        .background_gradient(cmap='Reds',subset=["deaths"])\
                        .background_gradient(cmap='Greens',subset=["recovered"])\
                        .background_gradient(cmap='Blues',subset=["active"])\
                        .background_gradient(cmap='Purples',subset=["death_rate (per 100)"])\
                        .background_gradient(cmap='Purples',subset=["recovery_rate (per 100)"])
barChart(statewise_cases , 'confirmed' , 'state' , 'Total Confirmed Cases' ,'h', color='confirmed' )
barChart(statewise_cases , 'deaths' , 'state' , 'Total Death Cases' ,'h', color='deaths' )
barChart(statewise_cases , 'recovered' , 'state' , 'Total Recovery Cases' ,'h', color='recovered' )
barChart(statewise_cases , 'active' , 'state' , 'Total Active Cases' ,'h', color='active')
barChart(statewise_cases , 'death_rate (per 100)' , 'state' , 'Death Rate (per 100)' ,'h', color='death_rate (per 100)' )
barChart(statewise_cases , 'recovery_rate (per 100)' , 'state' , 'Recovery Rate (per 100)' ,'h', color='recovery_rate (per 100)' )
def statelat(state):
    lat = {
        "Maharashtra":19.7515,
        "Delhi":28.7041,
        "Tamil Nadu":11.1271,
        "Rajasthan":27.0238,
        "Madhya Pradesh":22.9734,
        "Telangana":18.1124,
        "Gujarat":22.2587,
        "Uttar Pradesh":26.8467,
        "Andhra Pradesh":15.9129,
        "Kerala":10.8505,
        "Jammu and Kashmir":33.7782,
        "Karnataka":15.3173,
        "Haryana":29.0588,
        "Punjab":31.1471,
        "West Bengal":22.9868,
        "Bihar":25.0961,
        "Odisha":20.9517,
        "Uttarakhand":30.0668,
        "Himachal Pradesh":31.1048,
        "Assam":26.2006,
        "Chhattisgarh":22.0797,
        "Chandigarh":30.7333,
        "Jharkhand":23.6102,
        "Ladakh":34.152588,
        "Andaman and Nicobar Islands":11.7401,
        "Goa":15.2993,
        "Puducherry":11.9416,
        "Manipur":24.6637,
        "Tripura":23.9408,
        "Mizoram":23.1645,
        "Arunachal Pradesh":28.2180,
        "Dadra and Nagar Haveli and Daman and Diu":20.1809,
        "Nagaland":26.1584,
        "Daman and Diu":20.4283,
        "Lakshadweep":8.295441,
        "Meghalaya":25.4670,
        "Sikkim":27.5330,
        "State Unassigned":20.5937
    }
    return lat[state]
def statelong(state):
    long = {
        "Maharashtra":75.7139,
        "Delhi":77.1025,
        "Tamil Nadu":78.6569,
        "Rajasthan":74.2179,
        "Madhya Pradesh":78.6569,
        "Telangana":79.0193,
        "Gujarat":71.1924,
        "Uttar Pradesh":80.9462,
        "Andhra Pradesh":79.7400,
        "Kerala":76.2711,
        "Jammu and Kashmir":76.5762,
        "Karnataka":75.7139,
        "Haryana":76.0856,
        "Punjab":75.3412,
        "West Bengal":87.8550,
        "Bihar":85.3131,
        "Odisha":85.0985,
        "Uttarakhand":79.0193,
        "Himachal Pradesh":77.1734,
        "Assam":92.9376,
        "Chhattisgarh":82.1409,
        "Chandigarh":76.7794,
        "Jharkhand":85.2799,
        "Ladakh":77.577049,
        "Andaman and Nicobar Islands":92.6586,
        "Goa":74.1240,
        "Puducherry":79.8083,
        "Manipur":93.9063,
        "Tripura":91.9882,
        "Mizoram":92.9376,
        "Arunachal Pradesh":94.7278,
        "Dadra and Nagar Haveli and Daman and Diu":73.0169,
        "Nagaland":94.5624,
        "Daman and Diu":72.8397,
        "Lakshadweep":73.048973,
        "Meghalaya":91.3662,
        "Sikkim":88.5122,
        "State Unassigned":78.9629
    }
    return long[state]
len(statewise_cases)
a = {'states':list(statewise_cases['state']),
    'lat':list(statewise_cases['state'].apply(lambda x : statelat(x))),
    'long':list(statewise_cases['state'].apply(lambda x : statelong(x))),
    'confirmed':list(statewise_cases['confirmed']),
    'recovered':list(statewise_cases['recovered']),
    'deaths':list(statewise_cases['deaths'])}

df = pd.DataFrame.from_dict(a, orient='index')
dx = df.transpose()
dx.sample(20)
indiaMap = folium.Map(location=[23,80], tiles="Stamen Toner", zoom_start=4)

for lat, lon, value1,value2,value3, name in zip(dx['lat'], dx['long'], dx['confirmed'],dx['recovered'],dx['deaths'], dx['states']):
    folium.CircleMarker([lat, lon],
                        radius= (int((np.log(value1+1.00001))))*4,
                        popup = ('<strong>States</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed</strong>: ' + str(value1) + '<br>'),
                        color='#ff6600',
                        
                        fill_color='#ff8533',
                        fill_opacity=0.5 ).add_to(indiaMap)
    folium.CircleMarker([lat, lon],
                        radius= (int((np.log(value2+1.00001))))*4,
                        popup = ('<strong>States</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Recovered</strong>: ' + str(value2) + '<br>'),
                        color='#008000',
                        
                        fill_color='#008000',
                        fill_opacity=0.4 ).add_to(indiaMap)
    folium.CircleMarker([lat, lon],
                        radius= (int((np.log(value3+1.00001))))*4,
                        popup = ('<strong>States</strong>: ' + str(name).capitalize() + '<br>'
                                 '<strong>Confirmed</strong>: ' + str(value1) + '<br>'
                                 '<strong>Deaths</strong>: ' + str(value3) + '<br>'),
                        color='#0000A0',
                        
                        fill_color='#0000A0',
                        fill_opacity=0.4 ).add_to(indiaMap)
indiaMap
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

colors = ["#BF0A30", "#002868"]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)
#mask = np.array(Image.open("../input/circle/circle.png"))

notes = df_raw_data['notes']

text = " ".join(str(each) for each in notes)
stopwords = set(STOPWORDS)
stopwords.update(["Details", "awaited"])
wordcloud = WordCloud(stopwords=stopwords,max_words=100,colormap=cmap, background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()

df_daily_confirmed = pd.read_csv('http://api.covid19india.org/states_daily_csv/confirmed.csv')
df_daily_confirmed.head()
df_daily_decreased = pd.read_csv('https://api.covid19india.org/states_daily_csv/deceased.csv')
df_daily_decreased.head()
df_daily_recovered = pd.read_csv('https://api.covid19india.org/states_daily_csv/recovered.csv')
df_daily_recovered.head()

# removing the Unnamed column
df_daily_confirmed.drop(columns = 'Unnamed: 40' , axis=1,inplace=True )
df_daily_recovered.drop('Unnamed: 40',axis=1,inplace=True)
df_daily_decreased.drop('Unnamed: 40',axis=1,inplace=True)

#Getting daily sum of confirmed cases
df_daily_confirmed['Total_Confirmed_Cases'] = df_daily_confirmed.sum(axis=1)
df_daily_confirmed.head()

#Getting daily sum of Decreased cases
df_daily_decreased['Total_Decreased_Cases'] = df_daily_decreased.sum(axis=1)
df_daily_decreased.head()

#Getting daily sum of Recovered cases
df_daily_recovered['Total_Recovered_Cases'] = df_daily_confirmed.sum(axis=1)
df_daily_recovered.drop('Total_Recovered_Cases',axis=1,inplace=True)
df_daily_recovered.head()

#Getting daily sum of confirmed cases
df_daily_confirmed['Total_Confirmed_Cases'] = df_daily_confirmed.sum(axis=1)
df_daily_confirmed.head()

#Getting daily sum of confirmed cases
df_daily_decreased['Total_Decreased_Cases'] = df_daily_decreased.sum(axis=1)
df_daily_decreased.head()

#Getting daily sum of confirmed cases
df_daily_recovered['Total_Rcovered_Cases'] = df_daily_recovered.sum(axis=1)
df_daily_recovered.head()
fig_c = px.bar(df_daily_confirmed, x=df_daily_confirmed["date"], y=df_daily_confirmed["Total_Confirmed_Cases"], color_discrete_sequence = [cnf])
fig_d = px.bar(df_daily_decreased, x=df_daily_decreased["date"], y=df_daily_decreased['Total_Decreased_Cases'], color_discrete_sequence = [dth])

fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,
                    subplot_titles=('Confirmed cases', 'Deaths reported'))

fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)

fig.update_layout(height=480)
fig.show()
df_states_decreasedCount = pd.DataFrame(df_daily_decreased.sum(axis=0))
df_states_decreasedCount.columns = ['Total_death_count']
df_states_decreasedCount.drop('date').head()
# Data Preparation

df_daily_decreased['Deaths /100'] = np.round(100*df_daily_decreased['Total_Decreased_Cases']/df_daily_confirmed["Total_Confirmed_Cases"],2)
df_daily_recovered['Recovered / 100 Cases'] = np.round(100*df_daily_recovered['Total_Rcovered_Cases']/df_daily_confirmed['Total_Confirmed_Cases'],2)
df_daily_decreased['Death /100 Recovered'] = np.round(100*df_daily_decreased['Total_Decreased_Cases']/df_daily_recovered['Total_Rcovered_Cases'],2)

#plotting line chart
fig_1 = px.line(df_daily_decreased, x=df_daily_decreased["date"], y=df_daily_decreased['Deaths /100'], color_discrete_sequence = [dth])
fig_2 = px.line(df_daily_confirmed, x=df_daily_recovered["date"], y=df_daily_recovered['Recovered / 100 Cases'], color_discrete_sequence = [rec])
fig_3 = px.line(df_daily_decreased,x=df_daily_decreased["date"], y=df_daily_recovered['Recovered / 100 Cases'], color_discrete_sequence = ['#333333'])

fig = make_subplots(rows=1, cols=3, shared_xaxes=False, 
                    subplot_titles=('Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered'))

fig.add_trace(fig_1['data'][0], row=1, col=1)
fig.add_trace(fig_2['data'][0], row=1, col=2)
fig.add_trace(fig_3['data'][0], row=1, col=3)

fig.update_layout(height=480)
fig.show()
#data preparation
df_conf_dth_recovrd = df_daily_confirmed.filter(['date','Total_Confirmed_Cases'], axis=1)
df_conf_dth_recovrd['Total_Decreased_Cases'] = df_daily_decreased.filter(['Total_Decreased_Cases'], axis =1 )
df_conf_dth_recovrd['Total_Rcovered_Cases'] = df_daily_recovered.filter(['Total_Rcovered_Cases'], axis =1 )

df_conf_dth_recovrd.columns = ['ObservationDate', 'ConfirmedCases',"DeathReported",'RecoveredCases']
df_conf_dth_recovrd['DailyGrowthPercentagefromPreviousDay']=np.round(df_conf_dth_recovrd['ConfirmedCases'].pct_change(), 2)
df_conf_dth_recovrd['Active'] = df_conf_dth_recovrd['ConfirmedCases'] - (df_conf_dth_recovrd['DeathReported'] + df_conf_dth_recovrd['RecoveredCases'])
df_conf_dth_recovrd.head()
# plot of growth rate of confirmed cases
fig1 = px.scatter(df_conf_dth_recovrd, 
                 x='ObservationDate', 
                  y="DailyGrowthPercentagefromPreviousDay", 
                  text='DailyGrowthPercentagefromPreviousDay',
                  range_x=['2020-03-05','2020-04-25'])
fig1.update_traces(marker=dict(size=3,line=dict(width=2,color='DarkSlateGrey')),
                  marker_color=[dth],
                  mode='text+lines+markers',textposition='top center', )

fig1.update_layout( width=1500, height=900, title_text = '<b>Growth percent in number of total COVID-19 cases in India on each day compared to the previous day</b>')
fig1.show()
temp1 = df_conf_dth_recovrd.copy()
temp1 = temp1.melt(id_vars="ObservationDate", value_vars=['RecoveredCases', 'DeathReported', 'ConfirmedCases'],
                 var_name='Case', value_name='Count')
temp1.tail()

fig = px.area(temp1, x="ObservationDate", y="Count", color='Case', height=600,
           title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
#ploting spread over time
date_india_spread = df_conf_dth_recovrd.groupby('ObservationDate')['ConfirmedCases','DeathReported','RecoveredCases','Active'].sum().reset_index()

trace1 = go.Scatter(
                x=date_india_spread['ObservationDate'],
                y=date_india_spread['ConfirmedCases'],
                name="Confirmed",
                mode='lines+markers',
                line_color='orange')
trace2 = go.Scatter(
                x=date_india_spread['ObservationDate'],
                y=date_india_spread['DeathReported'],
                name="Deaths",
                mode='lines+markers',
                line_color='red')

trace3 = go.Scatter(
                x=date_india_spread['ObservationDate'],
                y=date_india_spread['RecoveredCases'],
                name="Recovered",
                mode='lines+markers',
                line_color='green')
trace4 = go.Scatter(
               x=date_india_spread['ObservationDate'],
               y=date_india_spread['Active'],
               name="Active",
                mode='lines+markers',
               line_color='blue')

layout = go.Layout(template="ggplot2", width=1200, height=500, title_text = '<b>Spread of the Coronavirus In India Over Time </b>')
fig = go.Figure(data = [trace1,trace2,trace3,trace4], layout = layout)
fig.show()

