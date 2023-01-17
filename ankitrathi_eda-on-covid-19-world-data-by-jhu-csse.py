# import the necessary libraries
import numpy as np 
import pandas as pd 
from datetime import date, timedelta

import requests
import io

# Visualisation libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
sdate = date.today()-timedelta(days=30)
dates = pd.date_range(sdate, periods=30, freq='D')
n=0
df = pd.DataFrame()
while n<=29:
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/' +dates[n].strftime('%m-%d-%Y')+ '.csv'
    tmp = pd.read_csv(url)
    df = df.append(tmp)
    n = n+1    
    
cdate = date.today()-timedelta(days=1)
df.head()
cols = ['Country/Region','Country_Region','Province/State','Last Update','Last_Update','Confirmed','Deaths','Recovered']
df["Country/Region"] = df["Country/Region"].combine_first(df["Country_Region"])
df["Last_Update"] = df["Last Update"].combine_first(df["Last_Update"])
cols = ['Country/Region','Province/State','Last_Update','Confirmed','Deaths','Recovered']
df = df[cols]
df["Last_Update"] = pd.to_datetime(df["Last_Update"]).dt.strftime('%Y-%m-%d')
df.head()
df.shape
# check missing values
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df
cols = ['Confirmed','Deaths','Recovered']
df[df["Last_Update"]==cdate.strftime('%Y-%m-%d')][cols].sum(axis = 0, skipna = True).astype(int)
# pie plot for current status
fig1, ax1 = plt.subplots()
#ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
df[df["Last_Update"]==cdate.strftime('%Y-%m-%d')][cols].sum(axis = 0, skipna = True).astype(int).plot(kind='pie', autopct='%1.1f%%', figsize=(10,5))
ax1.axis('equal')

#draw circle
centre_circle = plt.Circle((0,0),0.40,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Current Status: Overall', fontsize=14)
plt.show()
# bar plot for current status
tmp = df[df["Last_Update"]==cdate.strftime('%Y-%m-%d')].groupby(['Country/Region']).sum().sort_values(by=['Confirmed'], ascending=True)[-15:]
tmp.plot(kind='barh', figsize=(15,10))
plt.title('Current Status: Country-wise', fontsize=14)

plt.show()
# Detection State-wise
tmp = df.groupby(['Last_Update']).sum().sort_values(by=['Confirmed'], ascending=True)[-30:]
tmp.plot.area(stacked=False, figsize=(10,5))

plt.xticks(list(range(len(tmp.index))), tmp.index, fontsize=12, rotation=45)
plt.title('Status Date-wise', fontsize=14)
plt.show()
# Detection State-wise
tmp = df[df['Country/Region']=='China'].groupby(['Last_Update']).sum().sort_values(by=['Confirmed'], ascending=True)[-30:]
tmp.plot.area(stacked=False, figsize=(10,5))

plt.xticks(list(range(len(tmp.index))), tmp.index, fontsize=12, rotation=45)
plt.title('Status Date-wise: China', fontsize=14)
plt.show()
# Detection State-wise
tmp = df[df['Country/Region']=='Italy'].groupby(['Last_Update']).sum().sort_values(by=['Confirmed'], ascending=True)[-30:]
tmp.plot.area(stacked=False, figsize=(10,5))

plt.xticks(list(range(len(tmp.index))), tmp.index, fontsize=12, rotation=45)
plt.title('Status Date-wise: Italy', fontsize=14)
plt.show()
# Detection State-wise
tmp = df[df['Country/Region']=='US'].groupby(['Last_Update']).sum().sort_values(by=['Confirmed'], ascending=True)[-30:]
tmp.plot.area(stacked=False, figsize=(10,5))

plt.xticks(list(range(len(tmp.index))), tmp.index, fontsize=12, rotation=45)
plt.title('Status Date-wise: US', fontsize=14)
plt.show()
# Detection State-wise
tmp = df[df['Country/Region']=='Spain'].groupby(['Last_Update']).sum().sort_values(by=['Confirmed'], ascending=True)[-30:]
tmp.plot.area(stacked=False, figsize=(10,5))

plt.xticks(list(range(len(tmp.index))), tmp.index, fontsize=12, rotation=45)
plt.title('Status Date-wise: Spain', fontsize=14)
plt.show()
# Detection State-wise
tmp = df[df['Country/Region']=='Germany'].groupby(['Last_Update']).sum().sort_values(by=['Confirmed'], ascending=True)[-30:]
tmp.plot.area(stacked=False, figsize=(10,5))

plt.xticks(list(range(len(tmp.index))), tmp.index, fontsize=12, rotation=45)
plt.title('Status Date-wise: Germany', fontsize=14)
plt.show()
# Detection State-wise
tmp = df[df['Country/Region']=='Iran'].groupby(['Last_Update']).sum().sort_values(by=['Confirmed'], ascending=True)[-30:]
tmp.plot.area(stacked=False, figsize=(10,5))

plt.xticks(list(range(len(tmp.index))), tmp.index, fontsize=12, rotation=45)
plt.title('Status Date-wise: Iran', fontsize=14)
plt.show()
# Detection State-wise
tmp = df[df['Country/Region']=='France'].groupby(['Last_Update']).sum().sort_values(by=['Confirmed'], ascending=True)[-30:]
tmp.plot.area(stacked=False, figsize=(10,5))

plt.xticks(list(range(len(tmp.index))), tmp.index, fontsize=12, rotation=45)
plt.title('Status Date-wise: France', fontsize=14)
plt.show()
# Detection State-wise
tmp = df[df['Country/Region']=='India'].groupby(['Last_Update']).sum().sort_values(by=['Confirmed'], ascending=True)[-30:]
tmp.plot.area(stacked=False, figsize=(10,5))

plt.xticks(list(range(len(tmp.index))), tmp.index, fontsize=12, rotation=45)
plt.title('Status Date-wise', fontsize=14)
plt.show()