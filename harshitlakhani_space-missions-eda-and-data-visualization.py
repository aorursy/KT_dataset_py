#importing the modules necessary for this module

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt



import seaborn as sns

import squarify



import re
#reading the .csv file

raw_data = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')

raw_data.head()
#droped the unwanted columns from the dataframe

raw_data.drop(columns=['Unnamed: 0','Unnamed: 0.1'],inplace=True)

raw_data.head()
#changed some column names for better intuition

raw_data.rename(columns={'Datum':'Date-Time',' Rocket':'Cost'},inplace='True')

raw_data.sample(5)
#data information

raw_data.info()
#checking for the null values

raw_data.isna().sum()
#ploting the bar graph on missions conducted by all the companies

temp_data = raw_data['Company Name'].value_counts()

plt.figure(figsize=(25,10))

plt.bar(temp_data.index,temp_data.values)

plt.xticks(rotation=90)

plt.xlabel('Comapnies', weight='bold', fontsize=14)

plt.ylabel('Mission', weight='bold', fontsize=14)

plt.title('Missions per Company', weight='bold', fontsize=16)

plt.show()
#extracting the country name

countries = []

for location in raw_data.Location:

    countries.append(location.split(',')[-1])



raw_data['Country'] = np.array(countries)



raw_data.head(5)
#top 20 countries based on space missions conducted

top20_Country = raw_data['Country'].value_counts().head(20)



positive = []

negative = []



countries = raw_data.Country.value_counts()



#plot the data

plt.figure(figsize=(25,10))

plt.bar(countries.index,countries.values)

plt.xticks(rotation=45)

plt.xlabel('Countries', fontsize=14, weight='bold')

plt.ylabel('Missions', fontsize=14, weight='bold')

plt.title('Top 20 Countries', fontsize=16, weight='bold')

plt.show()

#ploting bar graph on mission conducted and the success rate of each of them

top20_Company = raw_data['Company Name'].value_counts().head(20)

positive = []

negative = []

labels = []

for company in top20_Company.index:

    unique_values = raw_data[raw_data['Company Name'] == company]['Status Mission'].value_counts()

    labels.append(int((unique_values[0]*100)/(unique_values[0]+unique_values[1])))

    positive.append(unique_values[0])

    negative.append(unique_values[1]) 

plt.figure(figsize=(25,10))

plt.bar(top20_Company.index,positive)

plt.bar(top20_Company.index,negative)



for i,v in enumerate(labels):

    plt.text(i-0.25,v/labels[i]+100,str(labels[i])+'%', fontsize=14)

    

plt.xticks(rotation=45)

plt.xlabel('Companies', weight='bold', fontsize=14)

plt.ylabel('Missions', weight='bold', fontsize=14)

plt.title('Top 20 companies', weight='bold', fontsize=16)

plt.legend(['Mission Sucess','Mission Failure'])

plt.show()
#treemap visualization of active missions

active_missions = raw_data[raw_data['Status Rocket'] == 'StatusActive']['Country'].value_counts().head(8)

values = active_missions.values[:]

index = active_missions.index[:]

colors = [plt.cm.Spectral(i/float(len(index))) for i in range(len(index))]



plt.figure(figsize=(25,10))

squarify.plot(label=index, sizes=values, color=colors, text_kwargs={'fontsize':14})

plt.title('Active Missions', weight='bold', fontsize=16)

plt.axis('off')



plt.show()
import re



#extracting the year of launch

years = []

for year in raw_data["Date-Time"]:

    years.append(re.split(r'\s',year)[3])



raw_data['Year'] = np.array(years)



time_series = {}

for country in [' USA',' China',' Russia']:

    time_series[country] =  raw_data[raw_data['Country'] == country]['Year'].value_counts().sort_index()

raw_data.head(5)
#top countries - based on their mission's every year

time_series = {}

for country in [' USA',' China',' Russia']:

    time_series[country] =  raw_data[raw_data['Country'] == country]['Year'].value_counts().sort_index()



plt.figure(figsize=(25,10))

plt.plot(time_series[' USA'].index, time_series[' USA'].values, label='USA')

plt.plot(time_series[' China'].index, time_series[' China'].values, label='China')

plt.plot(time_series[' Russia'].index, time_series[' Russia'].values, label='Russia')

plt.legend()

plt.xticks(rotation=90)

plt.xlabel('Year', fontsize=14, weight='bold')

plt.ylabel('Missions', fontsize=14, weight='bold')

plt.title('Missions per Year', fontsize=16, weight='bold')

plt.show()