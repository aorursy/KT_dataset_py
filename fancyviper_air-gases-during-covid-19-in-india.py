# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import os

import warnings

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from math import pi

import plotly.express as px

from IPython.display import HTML,display



warnings.filterwarnings("ignore")



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
city_day = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv")
city_day.shape
city_day
city_day.mean()
gases = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
gases


city_day = city_day.fillna(city_day.mean())

city_day.head()
pre_covid=city_day.loc[(city_day['Date']> '2014-12-31') & (city_day['Date']<'2020-0101')]

pre_covid=pre_covid.groupby("City").mean()

pre_covid.head()
pre_covid.shape
post_covid=city_day.loc[(city_day['Date']>'2019-12-31') & (city_day['Date']<'2020-06-01')]

post_covid=post_covid.groupby("City").mean()

post_covid.shape

P_So_2=sns.barplot(x=pre_covid.index,y=pre_covid['SO2'],palette="bright")

P_So_2.set_xticklabels(P_So_2.get_xticklabels(),rotation=90)

plt.title("Pre Covid SO2 levels in the atmosphere")
PO_So_2=sns.barplot(x=post_covid.index,y=post_covid['SO2'],palette='bright')

PO_So_2.set_xticklabels(PO_So_2.get_xticklabels(),rotation=90)

plt.title("Post Covid SO2 in atmosphere")
maincities=['Benguluru', 'Delhi', 'Mumbai', 'Patna']

maingases=['NH3' , 'O3' , 'NO2']

pre_covid_mod = pre_covid.loc[(pre_covid.index.isin(maincities))]

pre_covid_mod =pre_covid_mod[maingases]

print(pre_covid_mod)

pre_covid_mod = pre_covid.loc[(pre_covid.index.isin(maincities))]

pre_covid_mod =pre_covid_mod[maingases]

barWidth = 0.5

 

# set height of bar

bars1, bars2, bars3 = list(pre_covid_mod.NH3), list(pre_covid_mod.O3), list(pre_covid_mod.NO2)

 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

 

# Make the plot

plt.bar(r1, bars1, color='#994BBB', width=barWidth, edgecolor='black', label='NH3')

plt.bar(r2, bars2, color='#F12290', width=barWidth, edgecolor='black', label='O3')

plt.bar(r3, bars3, color='#F7E44A', width=barWidth, edgecolor='black', label='NO2')

# Add xticks on the middle of the group bars

plt.xlabel('group', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], list(pre_covid_mod.index))

 

# Create legend & Show graphic

plt.legend()

plt.show()
post_covid_mod = post_covid.loc[(post_covid.index.isin(maincities))]

post_covid_mod =post_covid_mod[maingases]



# set height of bar

bars1, bars2, bars3 = list(post_covid_mod.NH3), list(post_covid_mod.O3), list(post_covid_mod.NO2)

 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

 

# Make the plot

plt.bar(r1, bars1, color='#994BBB', width=barWidth, edgecolor='black', label='NH3')

plt.bar(r2, bars2, color='#F12290', width=barWidth, edgecolor='black', label='O3')

plt.bar(r3, bars3, color='#F7E44A', width=barWidth, edgecolor='black', label='NO2')

 

# Add xticks on the middle of the group bars

plt.xlabel('group', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], list(post_covid_mod.index))

 
plt.title("Gas Levels in different cities Pre-Covid")

plt.figure(figsize=(14,7))

# Heatmap showing average arrival delay for each airline by month

sns.heatmap(data=pre_covid, annot=False, cmap='inferno', fmt='g')
plt.title("Gas Levels in different cities Post-Covid")

plt.figure(figsize=(14,7))

# Heatmap showing average arrival delay for each airline by month

sns.heatmap(data=post_covid, annot=False, cmap='inferno', fmt='g')
df_2019 = city_day.loc[(city_day['Date'] > '2019-03-31') & (city_day['Date'] < '2019-04-15')]

df_2020 = city_day.loc[(city_day['Date'] > '2020-03-31') & (city_day['Date'] < '2020-04-15')]

compare_df = pd.concat([df_2019, df_2020])

compare_df.head()

compare_df.shape



df_mumbai = compare_df[compare_df['City'] == "Mumbai"]

df_mumbai[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'SO2', 'NH3', 'AQI', "Date"]].style.background_gradient(cmap='Blues')
df_delhi = compare_df[compare_df['City'] == "Delhi"]

df_delhi[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'SO2', 'NH3', 'AQI', "Date"]].style.background_gradient(cmap='Purples')
df_bang=compare_df[compare_df['City']=="Bengaluru"]

df_bang[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'SO2', 'NH3', 'AQI', "Date"]].style.background_gradient(cmap='Reds')
compare_df['Year'] = pd.DatetimeIndex(compare_df['Date']).year

compo3 = sns.catplot(x="City", y="O3", hue="Year", data = compare_df, height=10, aspect=1.8, kind="bar", palette="bright")

compo3.set_xticklabels(rotation=90)