# import the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, timedelta
from statsmodels.tsa.arima_model import ARIMA
from sklearn.cluster import KMeans
import matplotlib as mpl

from IPython.display import Markdown
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots
import plotly.express as px
import pycountry
import folium 
from folium import plugins

%config InlineBackend.figure_format = 'retina'
py.init_notebook_mode(connected=True)

# Utility Functions

'''Display markdown formatted output like bold, italic bold etc.'''
def formatted_text(string):
    display(Markdown(string))
# Import the data
nCoV_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

# Data Glimpse
nCoV_data.tail()
# Convert 'Last Update' column to datetime object
nCoV_data['Last Update'] = nCoV_data['Last Update'].apply(pd.to_datetime)
nCoV_data['ObservationDate'] = nCoV_data['ObservationDate'].apply(pd.to_datetime)

# Also drop the 'SNo' and the 'Date' columns
nCoV_data.drop(['SNo'], axis=1, inplace=True)

# Fill the missing values in 'Province/State' with the 'Country' name.
nCoV_data['Province/State'] = nCoV_data['Province/State'].replace(np.nan, nCoV_data['Country/Region'])

# Data Glimpse
nCoV_data.head()
# Lets rename the columns - 'Province/State' and 'Last Update' to remove the '/' and space respectively.
nCoV_data.rename(columns={'Last Update': 'LastUpdate', 'Province/State': 'State', 'Country/Region': 'Country', 'ObservationDate': 'Date'}, inplace=True)

# Data Glimpse
#nCoV_data.head()
# Active Case = confirmed - deaths - recovered
nCoV_data['Active'] = nCoV_data['Confirmed'] - nCoV_data['Deaths'] - nCoV_data['Recovered']

# Grouping on Basis of country since 
nCoV_data = nCoV_data.groupby(['Date','Country'])[['Confirmed','Deaths','Recovered','Active']].sum().reset_index()
nCoV_data['New_Cases'] = nCoV_data['Confirmed'] - nCoV_data.groupby(['Country'])['Confirmed'].shift(1)

# Check the Data Info again
#nCoV_data.info()
scb = pd.read_excel("../input/sc-regional-data/Countries.xlsx")
scb.rename(columns={'Countries':'Country'},inplace=True)                    
scb.head()
# Merge both the file and Create Final dataset for only SCB specific countries.

nCoV_scb = pd.merge(scb,nCoV_data,on='Country',how='inner',indicator=True)
nCoV_scb['Week_Day'] = nCoV_scb['Date'].dt.week
#nCoV_scb.head()
#Calculate day of year
nCoV_scb['Day_of_Year'] = nCoV_scb['Date'].dt.dayofyear

#nCoV_scb[nCoV_scb['Week_Day'].isnull()]

nCoV_scb['Day_of_Year'] = nCoV_scb['Date'].dt.dayofyear

nCoV_scb['Day_of_Year'] = nCoV_scb['Day_of_Year'].astype(int)
nCoV_Region = nCoV_scb.groupby(['Week_Day','Region','Date','Day_of_Year'])[['Confirmed','Deaths','Recovered','Active','New_Cases']].sum().reset_index()
nCoV_Country = nCoV_scb.groupby(['Region','Country','Week_Day','Date','Day_of_Year'])[['Confirmed','Deaths','Recovered','Active','New_Cases']].sum().reset_index()
c_list = nCoV_Country[nCoV_Country['Date'] == nCoV_Country.Date.max()]
c_list = c_list[c_list['Confirmed'] > 500]
list = c_list['Country']

nCoV_Country = nCoV_Country[nCoV_Country['Country'].isin(list)]
nCoV_Country.head()
# Lets check the total #Countries affected by nCoV

nCoV_Countries = nCoV_Country['Country'].unique().tolist()
print('\n')
print(nCoV_Countries)
print("\n------------------------------------------------------------------")
print("\n Total SCB Operated countries where cases are more than 500: ",len(nCoV_Countries))
# Create function to create Pie Chart & Bar Chart.

def pie_chart(df,type):
    nCoV_Region1 = df[df['Date'] == df.Date.max()]
    nCoV_Region1.sort_values('Confirmed',ascending=False,inplace=True)
    
    plt.figure(figsize=(10,5))
    colors = ['yellowgreen', 'gold', 'lightskyblue','red','blue','green','orange']
    sns.barplot(x=type,y='Confirmed',data=nCoV_Region1)
    
    plt.xlabel("Impacted Regions")
    plt.ylabel("Total Confrimed Cases")
    plt.show()
        
    plt.figure(figsize=(9,7))
    plt.pie(nCoV_Region1['Confirmed'],autopct='%1.1f%%',shadow=True,colors=colors)
    plt.legend(nCoV_Region1[type],loc='best')
    plt.show()

    return
# Create function for line plots
def regional(df,type,plots,count):
    nCoV_Region_sum = df[df['Confirmed'] > count]
    nCoV_Region_sum['Day_of_Year'] = nCoV_Region_sum.groupby(type)['Day_of_Year'].rank(method='min')
    
    fig, ax = plt.subplots()
    mpl.style.use('seaborn')
    
    plt.title("Standard Chartered : Daily Trend after 1000 cases", fontsize=15)

    nCoV_Region_sum.groupby(type).plot(x='Day_of_Year', y=plots, ax=ax,figsize=(10,6),linewidth=3)
    
    plt.xlabel("No of Days since 1000 Confrimed Cases")
    
    leg = nCoV_Region_sum.groupby(type)[plots].count().reset_index()
    plt.legend(leg[type],loc='best',fontsize=14)
        
    return
# Create function for Daily Cases.
def new_cases(df,type,plots,count):

    nCoV_Region_sum = df[df['Confirmed'] > count]
    nCoV_Region_sum['Day_of_Year'] = nCoV_Region_sum.groupby(type)['Day_of_Year'].rank(method='min').astype(int)
    
#    fig, ax = plt.subplots()
    mpl.style.use('seaborn')
    
    nCoV_Region_sum.plot(x='Day_of_Year',y='New_Cases',kind='bar',figsize=(12,6))
    plt.title("New Cases Day by Day ", fontsize=20)
    
    plt.xlabel("---Days since 100 Confrimed Cases-->")
    plt.ylabel("New Cases every day")
    
    plt.show()
        
    return
pd.options.mode.chained_assignment = None
pie_chart(nCoV_Region,'Region')
regional(nCoV_Region,'Region','Confirmed',1000)
plt.ylabel("Confrimed Cases(Cumilative)")
plt.show()
regional(nCoV_Region,'Region','Deaths',500)
plt.ylabel("Total Deaths (Cumilative)")
plt.show()

nCoV_Country1 = nCoV_Country[nCoV_Country['Date'] == nCoV_Country.Date.max()]

def sunbust():
    fig = px.sunburst(nCoV_Country1.sort_values(by='Active', ascending=False).reset_index(drop=True), 
                     path=["Region","Country"], values="Active", height=550,
                     title='Number of Active cases as of Date',
                     color_discrete_sequence = px.colors.qualitative.Prism)
    fig.data[0].textinfo = 'label+text+value'
    fig.show()
    return

sunbust()
#nCoV_Country1['Region'].unique()
nCoV_Country1 = nCoV_Country1[~nCoV_Country1['Region'].isin(['Europe','America'])]
nCoV_Country1['Region'].unique()

sunbust()
nCoV_ASA = nCoV_Country[nCoV_Country['Region']=='ASA']
pie_chart(nCoV_ASA,'Country')
regional(nCoV_ASA,'Country','Confirmed',500)
plt.ylabel("Confrimed Cases(Cumilative)")
plt.show()
regional(nCoV_ASA,'Country','Deaths',100)
plt.ylabel("Total Deaths(Cumilative)")
plt.show()
nCoV_ASA = nCoV_Region[nCoV_Region['Region']=='ASA']
new_cases(nCoV_ASA,'Region','New_Cases',500)
nCoV_GCNA = nCoV_Country[nCoV_Country['Region']=='GCNA']
pie_chart(nCoV_GCNA,'Country')
regional(nCoV_GCNA,'Country','Confirmed',500)
plt.ylabel("Confrimed Cases(Cumilative)")
plt.show()

regional(nCoV_GCNA,'Country','Deaths',100)
plt.ylabel("Total Deaths(Cumilative)")
plt.show()
nCoV_ASA = nCoV_Region[nCoV_Region['Region']=='GCNA']
new_cases(nCoV_ASA,'Region','New_Cases',500)
nCoV_Europe = nCoV_Country[nCoV_Country['Region']=='Europe']
pie_chart(nCoV_Europe,'Country')
regional(nCoV_Europe,'Country','Confirmed',500)
plt.ylabel("Confrimed Cases(Cumilative)")
plt.show()

regional(nCoV_Europe,'Country','Deaths',100)
plt.ylabel("Total Deaths(Cumilative)")
plt.show()

nCoV_Europe = nCoV_Region[nCoV_Region['Region']=='Europe']
new_cases(nCoV_Europe,'Region','New_Cases',500)
nCoV_ME = nCoV_Country[nCoV_Country['Region']=='ME']
pie_chart(nCoV_ME,'Country')
regional(nCoV_ME,'Country','Confirmed',500)
plt.ylabel("Confrimed Cases(Cumilative)")
plt.show()

regional(nCoV_ME,'Country','Deaths',100)
plt.ylabel("Total Deaths(Cumilative)")
plt.show()

nCoV_ME = nCoV_Region[nCoV_Region['Region']=='ME']
new_cases(nCoV_ME,'Region','New_Cases',100)
nCoV_Africa = nCoV_Country[nCoV_Country['Region']=='Africa']
pie_chart(nCoV_Africa,'Country')
regional(nCoV_Africa,'Country','Confirmed',500)
plt.ylabel("Confrimed Cases(Cumilative)")
plt.show()

regional(nCoV_Africa,'Country','Deaths',100)
plt.ylabel("Total Deaths(Cumilative)")
plt.show()

nCoV_ME = nCoV_Region[nCoV_Region['Region']=='Africa']
new_cases(nCoV_ME,'Region','New_Cases',500)
