



import numpy as np 

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

import conda

from wordcloud import WordCloud, STOPWORDS

from mpl_toolkits.basemap import Basemap



df = pd.read_csv('/kaggle/input/ufo-sightings/scrubbed.csv')
df['datetime'] = df['datetime'].str.replace('24:00', '0:00') #converts 24:00 to 0:00

df['month'] = pd.DatetimeIndex(df['datetime']).month #creates column 'month' stripping the year from datetime_zero

df['year'] = pd.DatetimeIndex(df['datetime']).year #creates column 'year' stripping the year from datetime_zero

df['day'] = pd.DatetimeIndex(df['datetime']).day #creates column 'day' stripping the year from datetime_zero

df['hour'] = pd.DatetimeIndex(df['datetime']).hour #creates column 'hour' stripping the year from datetime_zero

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
years_data = df['year'].value_counts()

years_index = years_data.index  

years_values = years_data.values

plt.figure(figsize=(15,8))

plt.xticks(rotation = 60)

plt.title('UFO Sightings since 1943', fontsize=18)

plt.xlabel("Year", fontsize=14)

plt.ylabel("Number of reports", fontsize=14)

years_plot = sns.barplot(x=years_index[:70],y=years_values[:70], palette = "RdPu_r")
def month_conversion(m):

    if m in range(3,6):

        return "Spring"

    if m in range(6,9):

        return "Summer"

    if m in range(9,12):

        return "Autumn"

    if m == 12 or m == 1 or m == 2:

        return "Winter"

    

season_order = ["Spring", "Summer", "Autumn", "Winter"]

    

df["Season"] = df['month'].apply(month_conversion)

season_stat = df["Season"].value_counts()

season_index = season_stat.index

season_value = season_stat.values

plt.figure(figsize=(15,8))

plt.title('UFO Sightings by Season', fontsize=18)

plt.xlabel("Season",fontsize=14)

plt.ylabel("Number of reports", fontsize=14)

season_plot = sns.barplot(x=season_index[:60],y=season_value[:60], palette = "Accent_r", alpha=0.3, order=season_order)
order = ["January", "February", "March", "April", "May", "June", "July","August", "September", 

         "October", "November", "December"]

df['month'] = df['datetime'].dt.month_name() #turns month numbers into month names

month_data = df['month'].value_counts() 

month_index = month_data.index 

month_values = month_data.values

plt.figure(figsize=(15,8))

plt.title('UFO Sightings by Month', fontsize=18)

plt.xlabel("Month",fontsize=14)

plt.ylabel("Number of reports", fontsize=14)

month_plot = sns.barplot(x=month_index[:60],y=month_values[:60], palette = "Blues", order=order)
day_data = df['day'].value_counts() 

day_index = day_data.index 

day_values =day_data.values

plt.figure(figsize=(15,8))

plt.title('UFO Sightings by Day', fontsize=18)

plt.xlabel("Day of the Month", fontsize=14)

plt.ylabel("Number of reports", fontsize=14)

day_plot = sns.barplot(x=day_index[:60],y=day_values[:60], palette = "Greens")
df['weekday'] = df['datetime'].dt.day_name()

order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

weekday_data = df['weekday'].value_counts()

weekday_index = weekday_data.index 

weekday_values = weekday_data.values

plt.figure(figsize=(15,8))

plt.title('UFO Sightings by Day of the Week', fontsize=18)

plt.xlabel("Day of the week", fontsize=14)

plt.ylabel("Number of reports", fontsize=14)

sns.set_palette("pastel")

weekday_plot = sns.barplot(x=weekday_index[:60],y=weekday_values[:60], order = order)
hour_data = df['hour'].value_counts() 

hour_index = hour_data.index 

hour_values =hour_data.values

plt.figure(figsize=(15,8))

plt.title('UFO Sightings by Hour', fontsize=18)

plt.xlabel("Time", fontsize=14)

plt.ylabel("Number of reports", fontsize=14)

hour_plot = sns.barplot(x=hour_index[:60],y=hour_values[:60], palette = 'Purples')
df = df.rename(columns={'longitude ':'longitude'}) #correcting typo in the dataframe

df.at[43782,'latitude']= 33.2001  #correcting another typo

df["latitude"] = df.latitude.astype(float)  #setting latitude as a real number
lat = df['latitude'].tolist()

lon = df['longitude'].tolist()



plt.figure(figsize=(30,20))

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\

            llcrnrlon=-180,urcrnrlon=180,lat_ts=50,resolution='c')

m.drawcoastlines()

m.fillcontinents(color='white',lake_color='lightblue', zorder=1)



# draw parallels and meridians.



m.drawmapboundary(fill_color="lightblue")

m.scatter(lon,lat,latlon=True, s=50, c='red', marker='.', alpha=1, edgecolor='b', linewidth=0.5, zorder=10)





plt.title("Geographical Distribution", size=24)

plt.show()

plt.subplots(figsize=(18,8))

expl = (0.1,0.05,0.2,0.4,0.8)

colors = ["cornflowerblue", "lightcoral", "lightgreen", "gold", "black"]

labels =['United States', 'Canada', 'United Kingdom', 'Australia', 'Germany']



df['country'].value_counts().plot(kind='pie',fontsize = 12, colors=colors, 

                                  explode=expl, figsize = (8,8), autopct='%.1f%%', 

                    pctdistance=1.15, labels=None)

plt.legend(labels=labels, loc='upper right')

plt.title('UFO Sightings by Country', size=24)



plt.xticks(rotation=45, fontsize=15)

plt.show()
usa_stats = (df['country']=='us')

usdf = df[usa_stats]



state_stats = usdf.state.value_counts()

state_index = state_stats.index 

state_values = state_stats.values

plt.figure(figsize=(15,8))

plt.title('UFO Sightings by US State', fontsize=24)

plt.xlabel("State", fontsize=14)

plt.ylabel("Number of reports", fontsize=14)

plt.xticks(rotation = 60, size=12)

state_plot = sns.barplot(x=state_index[:60],y=state_values[:60], palette='RdBu_r')
lat_us = usdf['latitude'].tolist()

lon_us = usdf['longitude'].tolist()





plt.figure(figsize=(30,20))



mapus = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

shpinfo = mapus.readshapefile('../input/datashp/cb_2018_us_state_500k', name='states', drawbounds=True)





mapus.drawmapboundary(fill_color='lightblue')

mapus.fillcontinents(color='white',lake_color='lightblue')

mapus.drawcoastlines()



mapus.scatter(lon_us,lat_us,latlon=True, s=50, c='lightgreen', marker='.', alpha=1, edgecolor='r', linewidth=0.5, zorder=10)

plt.title("Geographical Distribution in the US", size=24, alpha=0.7)



plt.show()
cmt = [item for item in df.comments.dropna()]

    

cmt = " ".join(cmt)



plt.figure(figsize=(18,12))



wordcloud = WordCloud(background_color='whitesmoke', width=2000, height=1000,

                      stopwords=None).generate(cmt)

plt.imshow(wordcloud, interpolation="nearest", aspect='auto')

plt.axis('off')

plt.savefig('wordcloud.png')

plt.title("Comment Wordcloud", size=40)



plt.show()
cmtus = [item for item in usdf.comments.dropna()]

    

cmtus = " ".join(cmtus)



plt.figure(figsize=(18,12))



wordcloud = WordCloud(background_color='black', width=2000,height=1000,stopwords=None).generate(cmtus)

plt.imshow(wordcloud, interpolation="nearest", aspect='auto')

plt.axis('off')

plt.savefig('wordcloudus.png')

plt.title("US Comment Wordcloud", size=40)



plt.show()
shape = df['shape'].value_counts()

shape_index = shape.index

shape_values = shape.values

plt.figure(figsize=(18,8))

shape = sns.barplot(x=shape_index[:60],y=shape_values[:60])

plt.xticks(rotation = 60)

plt.title("UFO Sightings by Shape", size=22)

plt.xlabel("Shape", fontsize=14)

plt.ylabel("Number of reports", fontsize=14)



#plt.legend(labels=labs, loc='upper right', bbox_to_anchor=(1.2, 1))

plt.show()
plt.subplots(figsize=(20,7))

duration_seconds = df["duration (seconds)"].value_counts()

dur_order = ["1 - 15","15 - 30", "30 - 60", "60 - 120","120 - 140", ">140" ]

duration_list=[]

for i in duration_seconds:

    if i in range(1,16):

        duration_list.append("1 - 15")

    if i in range(15,31):

        duration_list.append("15 - 30")

    if i in range(30,61):

        duration_list.append("30 - 60")

    if i in range(60,121):

        duration_list.append("60 - 120")

    if i in range(120,141):

        duration_list.append("120 - 140")

    if i > 140:

        duration_list.append(">140")



duration_list = pd.Series(duration_list)

duration_list = duration_list.value_counts()

dura_index = duration_list.index

dura_value = duration_list.values

duration = sns.barplot(dura_index, dura_value, order=dur_order)

plt.xlabel("Seconds", size=14)

plt.ylabel("Number of reports", size=14)

plt.title("UFO Sightings Duration")



plt.show()