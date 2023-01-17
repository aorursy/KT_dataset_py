import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from nltk.corpus import stopwords
from textblob import TextBlob
import datetime as dt
import warnings
import string
import time
# stop_words = []
stop_words = list(set(stopwords.words('english')))
warnings.filterwarnings('ignore')
punctuation = string.punctuation
df=pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
df.head()
df['participant_age_group'].head()
df.shape
df.isnull().sum()
df.dtypes
df['gun_type_parsed'] = df['gun_type'].fillna('0:Unknown')
gt = df.groupby(by=['gun_type_parsed']).agg({'n_killed': 'sum', 'n_injured' : 'sum', 'state' : 'count'}).reset_index().rename(columns={'state':'count'})

results = {}
for i, each in gt.iterrows():
    wrds = each['gun_type_parsed'].split("||")
    for wrd in wrds:
        if "Unknown" in wrd:
            continue
        wrd = wrd.replace("::",":").replace("|1","")
        gtype = wrd.split(":")[1]
        if gtype not in results: 
            results[gtype] = {'killed' : 0, 'injured' : 0, 'used' : 0}
        results[gtype]['killed'] += each['n_killed']
        results[gtype]['injured'] +=  each['n_injured']
        results[gtype]['used'] +=  each['count']

gun_names = list(results.keys())
used = [each['used'] for each in list(results.values())]
killed = [each['killed'] for each in list(results.values())]
injured = [each['injured'] for each in list(results.values())]
danger = []
for i, x in enumerate(used):
    danger.append((killed[i] + injured[i]) / x)

trace1 = go.Bar(x=gun_names, y=used, name='SF Zoo', orientation = 'v',
    marker = dict(color = '#EEE8AA', 
        line = dict(color = '#EEE8AA', width = 1) ))
data = [trace1]
layout = dict(height=400, title='Which guns have been used?', legend=dict(orientation="h"));
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='marker-h-bar')
df['n_guns'] = df['n_guns_involved'].apply(lambda x : "10+" if x>=10 else str(x))
# df['n_guns'].value_counts()
tempdf = df['n_guns'].value_counts().reset_index()
tempdf = tempdf[tempdf['index'] != 'nan']


labels = list(tempdf['index'])
values = list(tempdf['n_guns'])

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1', '#c0d1ed', '#efaceb', '#f5f794', '#94f794', '#fcc771']))
layout = dict(height=500, title='Number of Guns Used', legend=dict(orientation="v"));
fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)
df['date']=pd.to_datetime(df['date'])
df['year']=df['date'].dt.year
df_year=df.groupby(['year'])['n_killed','n_injured'].agg('sum')
states_df = df['state'].value_counts()

statesdf = pd.DataFrame()
statesdf['state'] = states_df.index
statesdf['counts'] = states_df.values

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

state_to_code = {'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}
statesdf['state_code'] = statesdf['state'].apply(lambda x : state_to_code[x])

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = statesdf['state_code'],
        z = statesdf['counts'],
        locationmode = 'USA-states',
        text = statesdf['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Gun Violence Incidents")
        ) ]

layout = dict(
        title = 'State wise number of Gun Violence Incidents',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )
df_year.plot(figsize=(10,8))
df_year.head(1)
df_year=df_year.reset_index()
fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df_year["year"], y=df_year['n_killed'], data=df_year)
f.set_xlabel("Year",fontsize=15)
f.set_ylabel("No. of Deaths",fontsize=15)
f.set_title('Year-wise Deaths due to Shootings')
for item in f.get_xticklabels():
    item.set_rotation(90)
fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df_year["year"], y=df_year['n_injured'], data=df_year)
f.set_xlabel("Year",fontsize=15)
f.set_ylabel("No. of Injuries",fontsize=15)
f.set_title('Year-wise Injuries due to Shootings')
for item in f.get_xticklabels():
    item.set_rotation(90)
temp = df["state"].value_counts().head(20)
trace = go.Bar(
    x=temp.index,
    y=temp.values,
    marker=dict(
        color=temp.values,
        colorscale = 'Picnic'
    ),
)

layout = go.Layout(
    title='Top 20 states with max. Gun Violence'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")
df.head(1)
def separate(df):
    df=df.split("||")
    df=[(x.split("::")) for x in df]
    y = []
    for  i in range (0, len(df)):
        y.append(df[i][-1])
    return(y) 
df['participant_gender'] = df['participant_gender'].fillna("0::Zero")
df['gender'] = df['participant_gender'].apply(lambda x: separate(x))
df['Males'] = df['gender'].apply(lambda x: x.count('Male'))
df['Females'] = df['gender'].apply(lambda x: x.count('Female'))
df.head(1)
dx=df[['state', 'Males', 'Females']].groupby('state').sum()
dx
trace1 = go.Bar(
    x=dx.index,
    y=dx['Males'],
    name='Males'
)
trace2 = go.Bar(
    x=dx.index,
    y=dx['Females'],
    name='Females'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title="Gender Ratio of Shooters"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
df_age=df.dropna(subset=['participant_age'])
df_age['ages'] = df_age['participant_age'].apply(lambda x: separate(x))
df_age=df_age[['ages','participant_age']]
df_age = pd.DataFrame(df_age.ages.values.tolist(), index= df_age.index)
df_age.head(1)
def separate_age(df):
    df=df.split("||")
    df=[(x.split("::")) for x in df]
    y = []
    for  i in range (0, len(df)):
        y.append(df[i][-1])
    return(y) 
df.head(1)
df_18=df.ix[df['year']==2018]
df_ts=df_18[['n_killed','date']]
df_ts.index=df_18['date']
df_ts['n_killed'].plot(figsize=(15,6), color="green")
plt.xlabel('Year')
plt.ylabel('No. of Deaths')
plt.title("Death due to Gun Violence Time-Series Visualization")
plt.show()
from fbprophet import Prophet
sns.set(font_scale=1) 
df_date_index = df_18[['date','n_killed']]
df_date_index = df_date_index.set_index('date')
df_prophet = df_date_index.copy()
df_prophet.reset_index(drop=False,inplace=True)
df_prophet.columns = ['ds','y']

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=270,freq='D')
forecast = m.predict(future)
fig = m.plot(forecast)
m.plot_components(forecast);
