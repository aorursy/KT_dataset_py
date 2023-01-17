import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas_profiling import ProfileReport
%matplotlib inline
#reading in the data
data = pd.read_csv('../input/air-quality-data-in-india/city_day.csv')
data.head(2)
# creating a new year column
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
report = ProfileReport(data)
report
sns.set_style("darkgrid")
sns.kdeplot(data=data['AQI'],label="AQI" ,shade=True)
aqi = data.groupby('City')['AQI'].min().reset_index()
aqi  = aqi.sort_values("AQI")
aqi = aqi.head(10)
fig1, ax1 = plt.subplots(figsize=(15,10))
ax1.pie(aqi['AQI'].tolist(), labels=aqi['City'].tolist(), autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend(loc='right',bbox_to_anchor=(1.2,0.9))
plt.show()
perc = data.loc[:,["year","City",'AQI']]
perc['mean_AQI'] = perc.groupby([perc.City,perc.year])['AQI'].transform('mean')
perc.drop('AQI', axis=1, inplace=True)
perc = perc.drop_duplicates()
perc = perc.sort_values("year",ascending = False)
top_brand = ['Talcher','Amritsar','Brajrajnagar'] 
perc = perc.loc[perc['City'].isin(top_brand)]
perc = perc.sort_values("year")
perc = perc.fillna(100)
fig=px.bar(perc,x='City', y="mean_AQI", animation_frame="year", 
           animation_group="City", color="City", hover_name="City")
fig.show()
aqi = data.groupby('City')['AQI'].max().reset_index()
aqi  = aqi.sort_values("AQI")
aqi = aqi.tail(10)
fig1, ax1 = plt.subplots(figsize=(15,10))
ax1.pie(aqi['AQI'].tolist(), labels=aqi['City'].tolist(), autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend(loc='right',bbox_to_anchor=(1.2,0.9))
plt.show()
perc = data.loc[:,["year","City",'AQI']]
perc['mean_AQI'] = perc.groupby([perc.City,perc.year])['AQI'].transform('mean')
perc.drop('AQI', axis=1, inplace=True)
perc = perc.drop_duplicates()
perc = perc.sort_values("year",ascending = False)
top_brand = ['Hyderabad','Amritsar','Gurugram','Guwahati',"Ahmedabad"] 
perc = perc.loc[perc['City'].isin(top_brand)]
perc = perc.sort_values(by="year")
perc = perc.fillna(100)
fig=px.bar(perc,x='City', y="mean_AQI", animation_frame="year", 
           animation_group="City", color="City", hover_name="City")
fig.show()
data.head()
data1 = data['AQI'].dropna()
top_10_city = data.loc[data1.index].groupby('City')['AQI'].mean().reset_index()
top_10_city.sort_values('AQI', ascending=False, inplace=True)
top_10_city.head(10)
top_cities = top_10_city.head(10)['City'].tolist()
top_cities
talcher = data[data['City']=='Talcher']
data_by_year = talcher.groupby('year')['AQI'].mean().reset_index().dropna()
data_by_year.head()
plt.plot(data_by_year['year'], data_by_year['AQI'])
plt.xticks(data_by_year['year'].tolist())
plt.title('Year wise mean AQI for Talcher')
plt.xlabel('Years')
plt.ylabel('Mean AQI')
plt.show()
braj = data[data['City']=='Brajrajnagar']
data_by_year = braj.groupby('year')['AQI'].mean().reset_index().dropna()
data_by_year.head()
plt.plot(data_by_year['year'], data_by_year['AQI'])
plt.xticks(data_by_year['year'].tolist())
plt.title('Year wise mean AQI for Talcher')
plt.xlabel('Years')
plt.ylabel('Mean AQI')
plt.show()
fig = plt.figure(figsize=(15,28))
for city,num in zip(top_cities, range(1,11)):
    df = data[data['City']==city]
    data_by_year = df.groupby('year')['AQI'].mean().reset_index().dropna()
    ax = fig.add_subplot(5,2,num)
    ax.plot(data_by_year['year'], data_by_year['AQI'])
    ax.set_xticks(data_by_year['year'].tolist())
    ax.set_title('Year wise mean AQI for {}'.format(city))
    ax.set_ylabel('Mean AQI')
data.head(2)
#imputing null values with 0.0
df = data.fillna(0.0)
no = df.groupby('City')['NO'].mean().reset_index()
no  = no.sort_values("NO")
no = no.head(10)
fig1, ax1 = plt.subplots(figsize=(15,10))
ax1.pie(no['NO'].tolist(), labels=no['City'].tolist(), autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend(loc='right',bbox_to_anchor=(1.2,0.9))
plt.show()
perc = df.loc[:,["year","City",'NO']]
perc['mean_NO'] = perc.groupby([perc.City,perc.year])['NO'].transform('mean')
perc.drop('NO', axis=1, inplace=True)
perc = perc.drop_duplicates()
perc = perc.sort_values("year",ascending = False)
top_brand = ['Hyderabad','Bhopal','Jorapokhar','Chennai',"Bengaluru"] 
perc = perc.loc[perc['City'].isin(top_brand)]
perc = perc.sort_values("year")
perc = perc.fillna(100)
fig=px.bar(perc,x='City', y="mean_NO", animation_frame="year", 
           animation_group="City", color="City", hover_name="City")
fig.show()
fig = plt.figure(figsize=(15,23))
for city,num in zip(top_cities, range(1,11)):
    df = data[data['City']==city]
    df = df.groupby('year')['NO'].mean().reset_index().dropna()
    ax = fig.add_subplot(5,2,num)
    ax.set_title(city)
    sns.kdeplot(data=df['NO'],label="NO" ,shade=True)
data.head(2)
sns.scatterplot('PM2.5', 'AQI', hue='year', data=data)
plt.title('Relation between PM2.5 and AQI')
plt.xlabel('PM2.5')
plt.ylabel('AQI')
plt.show()
variables = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']

fig = plt.figure(figsize=(16,34))
for variable,num in zip(variables, range(1,len(variables)+1)):
    ax = fig.add_subplot(6,2,num)
    sns.scatterplot(variable, 'AQI', hue='year', data=data)
    plt.title('Relation between {} and AQI'.format(variable))
    plt.xlabel(variable)
    plt.ylabel('AQI')
cities = ['Delhi', 'Ahmedabad', 'Hyderabad', 'Bengaluru', 'Kolkata']
fig,ax = plt.subplots(figsize=(15, 7))

for city in cities: 
    sns.lineplot(x="Date", y="AQI", data=data[data['City']==city].iloc[::30],label = city)

ax.set_xticklabels(ax.get_xticklabels(cities), rotation=30, ha="left")

ax.set_title('AQI values in cities')
ax.legend()
