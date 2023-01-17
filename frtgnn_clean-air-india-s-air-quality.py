import os

import warnings

import numpy as np

import pandas as pd

from math import pi

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

from IPython.display import HTML,display



warnings.filterwarnings("ignore")



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_station_hour = pd.read_csv("/kaggle/input/air-quality-data-in-india/station_hour.csv")

df_city_hour    = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_hour.csv")

df_station_day  = pd.read_csv("/kaggle/input/air-quality-data-in-india/station_day.csv")

df_city_day     = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv")

df_stations     = pd.read_csv("/kaggle/input/air-quality-data-in-india/stations.csv")
df_station_hour.head()
df_station_hour['Datetime'].min() , df_station_hour['Datetime'].max()
df_station_day.head()
df_station_day['Date'].min() , df_station_day['Date'].max()
df_city_day.head()
df_city_day['Date'].min() , df_city_day['Date'].max()
df_stations.head()
!pip install bar_chart_race
df_chart = df_city_day.pivot(index='Date', columns='City', values='CO')

df_chart = df_chart.fillna(df_chart.mean())

df_chart.head()
import bar_chart_race as bcr



df_chart_simplified = df_chart.iloc[::2]



bcr_html = bcr.bar_chart_race(df=df_chart_simplified,

                              filename=None,

                              orientation='h',

                              sort='desc',

                              n_bars=10,

                              label_bars=True,

                              use_index=True,

                              steps_per_period=10,

                              period_length=500,

                              figsize=(4, 3.5),

                              cmap='dark24',

                              title='City-Wise CO Pollution Levels ',

                              bar_label_size=7,

                              tick_label_size=7,

                              period_label_size=16,

                              fig=None)



display(HTML(bcr_html))

#thank you Ted , https://github.com/dexplo/bar_chart_race
df_city_day  = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv")

df_city_day  = df_city_day.fillna(df_city_day.mean())



df_Ahmedabad = df_city_day[df_city_day['City']== 'Ahmedabad']

df_Bengaluru = df_city_day[df_city_day['City']== 'Bengaluru']

df_Delhi     = df_city_day[df_city_day['City']== 'Delhi']

df_Hyderabad = df_city_day[df_city_day['City']== 'Hyderabad']

df_Kolkata   = df_city_day[df_city_day['City']== 'Kolkata']

df_Kolkata.head()
fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 3})



sns.lineplot(x="Date", y="CO", data=df_Ahmedabad.iloc[::30], color="b",label = 'Ahmedabad')

sns.lineplot(x="Date", y="CO", data=df_Kolkata.iloc  [::30], color="r",label = 'Kolkata')

sns.lineplot(x="Date", y="CO", data=df_Bengaluru.iloc[::30], color="g",label = 'Bengaluru')

sns.lineplot(x="Date", y="CO", data=df_Delhi    .iloc[::30], color="y",label = 'Delhi    ')

sns.lineplot(x="Date", y="CO", data=df_Hyderabad.iloc[::30], color="black",label = 'Hyderabad')



labels = [item.get_text() for item in ax.get_xticklabels()]

labels[1] = 'January 2015 to April 2020'

ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('CO LEVEL FROM DIFFERENT CITIES')

ax.legend(fontsize = 14)



fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 3})



sns.lineplot(x="Date", y="NO", data=df_Ahmedabad.iloc[::30], color="b",label = 'Ahmedabad')

sns.lineplot(x="Date", y="NO", data=df_Kolkata.iloc  [::30], color="r",label = 'Kolkata')

sns.lineplot(x="Date", y="NO", data=df_Bengaluru.iloc[::30], color="g",label = 'Bengaluru')

sns.lineplot(x="Date", y="NO", data=df_Delhi    .iloc[::30], color="y",label = 'Delhi    ')

sns.lineplot(x="Date", y="NO", data=df_Hyderabad.iloc[::30], color="black",label = 'Hyderabad')



ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('NO LEVEL FROM DIFFERENT CITIES')

ax.legend(fontsize = 14)



fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 3})



sns.lineplot(x="Date", y="PM10", data=df_Ahmedabad.iloc[::30], color="b",label = 'Ahmedabad')

sns.lineplot(x="Date", y="PM10", data=df_Kolkata.iloc  [::30], color="r",label = 'Kolkata')

sns.lineplot(x="Date", y="PM10", data=df_Bengaluru.iloc[::30], color="g",label = 'Bengaluru')

sns.lineplot(x="Date", y="PM10", data=df_Delhi    .iloc[::30], color="y",label = 'Delhi    ')

sns.lineplot(x="Date", y="PM10", data=df_Hyderabad.iloc[::30], color="black",label = 'Hyderabad')



ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('PM10 LEVEL FROM DIFFERENT CITIES')

ax.legend(fontsize = 14)



fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 3})



sns.lineplot(x="Date", y="NO2", data=df_Ahmedabad.iloc[::30], color="b",label = 'Ahmedabad')

sns.lineplot(x="Date", y="NO2", data=df_Kolkata.iloc  [::30], color="r",label = 'Kolkata')

sns.lineplot(x="Date", y="NO2", data=df_Bengaluru.iloc[::30], color="g",label = 'Bengaluru')

sns.lineplot(x="Date", y="NO2", data=df_Delhi    .iloc[::30], color="y",label = 'Delhi    ')

sns.lineplot(x="Date", y="NO2", data=df_Hyderabad.iloc[::30], color="black",label = 'Hyderabad')



ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('NO2 LEVEL FROM DIFFERENT CITIES')

ax.legend(fontsize = 14)



fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 1})



sns.barplot(x="Date", y="NOx", data=df_Ahmedabad.iloc[::30], color="b",label = 'Ahmedabad')

sns.barplot(x="Date", y="NOx", data=df_Kolkata.iloc  [::30], color="r",label = 'Kolkata')

sns.barplot(x="Date", y="NOx", data=df_Bengaluru.iloc[::30], color="g",label = 'Bengaluru')

sns.barplot(x="Date", y="NOx", data=df_Delhi    .iloc[::30], color="y",label = 'Delhi    ')

sns.barplot(x="Date", y="NOx", data=df_Hyderabad.iloc[::30], color="black",label = 'Hyderabad')



ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('NOx LEVEL FROM DIFFERENT CITIES')

ax.legend(fontsize = 14)



fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 1})



sns.barplot(x="Date", y="NH3", data=df_Delhi    .iloc[::30], color="y",label = 'Delhi    ')

sns.barplot(x="Date", y="NH3", data=df_Ahmedabad.iloc[::30], color="b",label = 'Ahmedabad')

sns.barplot(x="Date", y="NH3", data=df_Hyderabad.iloc[::30], color="black",label = 'Hyderabad')

sns.barplot(x="Date", y="NH3", data=df_Bengaluru.iloc[::30], color="g",label = 'Bengaluru')

sns.barplot(x="Date", y="NH3", data=df_Kolkata.iloc  [::30], color="r",label = 'Kolkata')





ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('NH3 LEVEL FROM DIFFERENT CITIES')

ax.legend(fontsize = 14);
fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})



sns.lineplot(x="Date", y="AQI", data=df_Delhi    .iloc[::30], color="y",label = 'Delhi    ')

sns.lineplot(x="Date", y="AQI", data=df_Ahmedabad.iloc[::30], color="b",label = 'Ahmedabad')

sns.lineplot(x="Date", y="AQI", data=df_Hyderabad.iloc[::30], color="black",label = 'Hyderabad')

sns.lineplot(x="Date", y="AQI", data=df_Bengaluru.iloc[::30], color="g",label = 'Bengaluru')

sns.lineplot(x="Date", y="AQI", data=df_Kolkata.iloc  [::30], color="r",label = 'Kolkata')





ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('AQI LEVEL FROM DIFFERENT CITIES')

ax.legend(fontsize = 14);
fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})



sns.distplot(df_Delhi    ["AQI"].iloc[::30], color="y",label = 'Delhi    ')

sns.distplot(df_Ahmedabad["AQI"].iloc[::30], color="b",label = 'Ahmedabad')

sns.distplot(df_Hyderabad["AQI"].iloc[::30], color="black",label = 'Hyderabad')

sns.distplot(df_Bengaluru["AQI"].iloc[::30], color="g",label = 'Bengaluru')

sns.distplot(df_Kolkata  ["AQI"].iloc  [::30], color="r",label = 'Kolkata')





ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('AQI DISTRIBUTION FROM DIFFERENT CITIES')

ax.legend(fontsize = 14)



fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})



sns.distplot(df_Delhi    ["CO"].iloc[::30], color="y",label = 'Delhi    ')

sns.distplot(df_Ahmedabad["CO"].iloc[::30], color="b",label = 'Ahmedabad')

sns.distplot(df_Hyderabad["CO"].iloc[::30], color="black",label = 'Hyderabad')

sns.distplot(df_Bengaluru["CO"].iloc[::30], color="g",label = 'Bengaluru')

sns.distplot(df_Kolkata  ["CO"].iloc  [::30], color="r",label = 'Kolkata')





ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('CO DISTRIBUTION FROM DIFFERENT CITIES')

ax.legend(fontsize = 14)



fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})



sns.distplot(df_Delhi    ["NOx"].iloc[::30], color="y",label = 'Delhi    ')

sns.distplot(df_Ahmedabad["NOx"].iloc[::30], color="b",label = 'Ahmedabad')

sns.distplot(df_Hyderabad["NOx"].iloc[::30], color="black",label = 'Hyderabad')

sns.distplot(df_Bengaluru["NOx"].iloc[::30], color="g",label = 'Bengaluru')

sns.distplot(df_Kolkata  ["NOx"].iloc  [::30], color="r",label = 'Kolkata')





ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('NOx DISTRIBUTION FROM DIFFERENT CITIES')

ax.legend(fontsize = 14);
df_CO = df_city_day.pivot(index='Date', columns='City', values='CO')

df_CO = df_CO.fillna(df_CO.mean())



df_NO = df_city_day.pivot(index='Date', columns='City', values='NO')

df_NO = df_NO.fillna(df_NO.mean())



df_SO2 = df_city_day.pivot(index='Date', columns='City', values='SO2')

df_SO2 = df_SO2.fillna(df_SO2.mean())



df_O3 = df_city_day.pivot(index='Date', columns='City', values='O3')

df_O3 = df_O3.fillna(df_O3.mean())



categories=list(df_CO)[0:]

N = len(categories)



values_co  = df_CO.mean(axis=0)

values_no  = df_NO.mean(axis=0)

values_so2 = df_SO2.mean(axis=0)

values_o3  = df_O3.mean(axis=0)



angles = [n / float(N-1) * 2 * pi for n in range(N-1)]

angles += angles[:1]



 

fig = plt.figure(figsize=(16,14))

ax = plt.subplot(111, polar=True)

 

plt.xticks(angles[:-1], categories, color='black', size=10)

 

ax.set_rlabel_position(0)

plt.yticks([0,5,10,15,20,25,30,35,40], ["0","5","10","15","20","25","30","35","40"], color="grey", size=12)

plt.ylim(0,40)

 

ax.plot(angles, values_co, 'red',linewidth=1, linestyle='solid', label="CO LEVELS")

ax.fill(angles, values_co, 'red', alpha=0.1)



ax.plot(angles, values_no, 'blue',linewidth=1, linestyle='solid', label="NO LEVELS")

ax.fill(angles, values_no, 'blue', alpha=0.1)



ax.plot(angles, values_so2,'green',linewidth=1, linestyle='solid', label="SO2 LEVELS")

ax.fill(angles, values_so2,'green', alpha=0.1)



ax.plot(angles, values_o3, 'yellow',linewidth=1, linestyle='solid', label="O3 LEVELS")

ax.fill(angles, values_o3, 'yellow', alpha=0.1)



plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title("Average Pollution Levels",fontsize=20);
df_CO.index = pd.to_datetime(df_CO.index)

df_CO_2019_04 = df_CO.loc['2019-04-01':'2019-04-10']

df_CO_2020_04 = df_CO.loc['2020-04-01':'2020-04-10']



df_CO_2019_04['Month'] = "03"

df_CO_2020_04['Month'] = "04"



df_CO_04 = pd.concat([df_CO_2019_04,df_CO_2020_04])

#df_CO_04.head()
sns.set(style="whitegrid")



g = sns.catplot(x="Month", y="Ahmedabad", data=df_CO_04,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("CO LEVEL")



plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title("Ahmedabad April 2019 vs 2020",fontsize=20);
df_city_day     = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv",index_col="Date")

df_city_day.fillna(df_city_day.mean(),inplace=True)

df_city_day.index = pd.to_datetime(df_city_day.index)



df_2019_04 = df_city_day.loc['2019-04-01':'2019-04-10']

df_2020_04 = df_city_day.loc['2020-04-01':'2020-04-10']



df_2019_04['Year'] = "2019"

df_2020_04['Year'] = "2020"



df_comparison = pd.concat([df_2019_04,df_2020_04])

#df_comparison.head()
df_comparison.tail()
chart = sns.catplot(x="City", y="CO", hue="Year", data=df_comparison, height=14, aspect=1.6, kind="bar", palette="muted");

chart.set_xticklabels(rotation=45);



chart = sns.catplot(x="City", y="NO", hue="Year", data=df_comparison, height=14, aspect=1.6, kind="bar", palette="muted");

chart.set_xticklabels(rotation=45);



chart = sns.catplot(x="City", y="O3", hue="Year", data=df_comparison, height=14, aspect=1.6, kind="bar", palette="muted");

chart.set_xticklabels(rotation=45);



chart = sns.catplot(x="City", y="Benzene", hue="Year", data=df_comparison, height=14, aspect=1.6, kind="bar", palette="muted");

chart.set_xticklabels(rotation=45);
df_city_day     = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv")

df_city_day.fillna(df_city_day.mean(),inplace=True)

df_city_day['Date'] = pd.to_datetime(df_city_day['Date'])



df_city_day.head()
df_ahmedabad = df_comparison[df_comparison['City'] == "Ahmedabad"]

df_ahmedabad[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'Xylene', 'AQI',"Year"]].style.background_gradient(cmap='Reds')
df_bengaluru = df_comparison[df_comparison['City'] == "Bengaluru"]

df_bengaluru[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'Xylene', 'AQI',"Year"]].style.background_gradient(cmap='Reds')
df_delhi = df_comparison[df_comparison['City'] == "Delhi"]

df_delhi[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'Xylene', 'AQI',"Year"]].style.background_gradient(cmap='Reds')
df_comparison.reset_index(inplace=True)
df_chart = df_comparison.pivot(index='Date', columns='City', values='AQI')

df_chart[['Ahmedabad','Amaravati', 'Bengaluru','Brajrajnagar', 'Chennai', 'Delhi', 'Gurugram', 'Hyderabad', 'Jaipur', 'Lucknow']][::1].style.background_gradient(cmap='Reds')