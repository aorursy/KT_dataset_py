# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing mandatory Libraries 

import numpy as np 

import pandas as pd 

import os



#Importing Mandatory Libraries for Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import pycountry 

import plotly.express as px 

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

!pip install chart_studio

import chart_studio.plotly as py

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



#Importing Libraries for changing the properties of default plot size 

plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")



#importing Libraries for Geographical plotting 

import folium 

from folium import Choropleth, Circle, Marker

from folium import plugins 

from folium.plugins import HeatMap, MarkerCluster



#Racing Bar Chart

!pip install bar_chart_race

import bar_chart_race as bcr

from IPython.display import HTML



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# For disabling Warning 

import warnings

warnings.filterwarnings('ignore')
station_hour=pd.read_csv('../input/air-quality-data-in-india/station_hour.csv')
station_hour
station_day=pd.read_csv('../input/air-quality-data-in-india/station_day.csv')

city_day=pd.read_csv('../input/air-quality-data-in-india/city_day.csv')

city_hour=pd.read_csv('../input/air-quality-data-in-india/city_hour.csv')

db_cities=pd.read_csv('../input/indian-cities-database/Indian Cities Database.csv')
#analysing Data 

city_day.isnull().sum()
db_cities.isnull().sum()
df_chart = city_day.pivot(index='Date', columns='City', values='CO')

df_chart = df_chart.fillna(df_chart.mean())

df_chart.head()
#data distribution city wise

city_day=pd.read_csv('../input/air-quality-data-in-india/city_day.csv')

city_day=city_day.fillna(city_day.mean())



df_Ahmedabad=city_day[city_day['City']=='Ahmedabad']

df_Bengaluru=city_day[city_day['City']=='Bengaluru']

df_Delhi=city_day[city_day['City']=='Delhi']

df_Hyderabad = city_day[city_day['City']== 'Hyderabad']

df_Kolkata   = city_day[city_day['City']== 'Kolkata']
df_Kolkata
fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 3})



sns.lineplot(x="Date", y="CO", data=df_Ahmedabad.iloc[::30], color="black",label = 'Ahmedabad')

sns.lineplot(x="Date", y="CO", data=df_Kolkata.iloc  [::30], color="b",label = 'Kolkata')

sns.lineplot(x="Date", y="CO", data=df_Bengaluru.iloc[::30], color="r",label = 'Bengaluru')

sns.lineplot(x="Date", y="CO", data=df_Delhi    .iloc[::30], color="g",label = 'Delhi    ')

sns.lineplot(x="Date", y="CO", data=df_Hyderabad.iloc[::30], color="y",label = 'Hyderabad')



labels = [item.get_text() for item in ax.get_xticklabels()]

labels[1] = 'Jan 2015 to Apr 2020'

ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15





ax.set_title('CO LEVEL FROM DIFFERENT CITIES')

ax.legend(fontsize = 14)



fig,ax = plt.subplots(figsize=(20, 10))

sns.despine(fig, left=True, bottom=True)

sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 3})



sns.lineplot(x="Date", y="NO", data=df_Ahmedabad.iloc[::30], color="black",label = 'Ahmedabad')

sns.lineplot(x="Date", y="NO", data=df_Kolkata.iloc  [::30], color="b",label = 'Kolkata')

sns.lineplot(x="Date", y="NO", data=df_Bengaluru.iloc[::30], color="r",label = 'Bengaluru')

sns.lineplot(x="Date", y="NO", data=df_Delhi    .iloc[::30], color="g",label = 'Delhi    ')

sns.lineplot(x="Date", y="NO", data=df_Hyderabad.iloc[::30], color="y",label = 'Hyderabad')



ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")

plt.rcParams["xtick.labelsize"] = 15



ax.set_title('LEVEL of NO for DIFFERENT CITIES')

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



ax.set_title('LEVEL of PM10 for DIFFERENT CITIES')

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
#CityWise Comparison of 10 Daya in 2019 & 2020

df_city_day     = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv")

df_city_day.fillna(df_city_day.mean(),inplace=True)

df_city_day['Date'] = pd.to_datetime(df_city_day['Date'])

                      
df_city_day
df_city_day     = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv",index_col="Date")

df_city_day.fillna(df_city_day.mean(),inplace=True)

df_city_day.index = pd.to_datetime(df_city_day.index)



df_2019_04 = df_city_day.loc['2019-04-01':'2019-04-10']

df_2020_04 = df_city_day.loc['2020-04-01':'2020-04-10']



df_2019_04['Year'] = "2019"

df_2020_04['Year'] = "2020"



df_comparison = pd.concat([df_2019_04,df_2020_04])
df_comparison.tail()
chart = sns.catplot(x="City", y="CO", hue="Year", data=df_comparison, height=14, aspect=1.6, kind="bar", palette="muted");

chart.set_xticklabels(rotation=45);



chart = sns.catplot(x="City", y="NO", hue="Year", data=df_comparison, height=14, aspect=1.6, kind="bar", palette="muted");

chart.set_xticklabels(rotation=45);



chart = sns.catplot(x="City", y="O3", hue="Year", data=df_comparison, height=14, aspect=1.6, kind="bar", palette="muted");

chart.set_xticklabels(rotation=45);



chart = sns.catplot(x="City", y="Benzene", hue="Year", data=df_comparison, height=14, aspect=1.6, kind="bar", palette="muted");

chart.set_xticklabels(rotation=45);
#Ahmedabad

df_Ahmedabad=df_comparison[df_comparison['City']=='Ahmedabad']

df_Ahmedabad[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'Xylene', 'AQI',"Year"]].style.background_gradient(cmap='Greens')
df_Kolkata=df_comparison[df_comparison['City']=='Kolkata']

df_Kolkata[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'Xylene', 'AQI',"Year"]].style.background_gradient(cmap='Reds')
df_Delhi=df_comparison[df_comparison['City']=='Delhi']

df_Delhi[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'Xylene', 'AQI',"Year"]].style.background_gradient(cmap='Blues')
df_comparison.head()
#Combining Major Components

df_city_day = pd.read_csv('../input/air-quality-data-in-india/city_day.csv')

df_city_day['Date'] = pd.to_datetime(df_city_day['Date'])

df_city_day['BTX'] = df_city_day['Benzene']+df_city_day['Toluene']+df_city_day['Xylene']

df_city_day.drop(['Benzene','Toluene','Xylene'],axis=1);

df_city_day['Particulate_Matter'] = df_city_day['PM2.5']+df_city_day['PM10']
pollutants = ['PM2.5','PM10','NO2', 'CO', 'SO2','O3', 'BTX']
df_city_day.set_index('Date',inplace=True)

axes = df_city_day[pollutants].plot(marker='.', alpha=0.5, linestyle='None', figsize=(16, 20), subplots=True)

for ax in axes:

    

    ax.set_xlabel('Years')

    ax.set_ylabel('ug / m3')
#Max Polluted Cities 

def max_polluted_city(pollutant):

    x1 = city_day[[pollutant,'City']].groupby(["City"]).mean().sort_values(by=pollutant,ascending=False).reset_index()

    x1[pollutant] = round(x1[pollutant],2)

    return x1[:10].style.background_gradient(cmap='OrRd')

from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.render()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

pm2_5 = max_polluted_city('PM2.5')

pm10 = max_polluted_city('PM10')

no2 = max_polluted_city('NO2')

so2 = max_polluted_city('SO2')

co = max_polluted_city('CO')







display_side_by_side(pm2_5,pm10,no2,so2,co)
#Patna, Delhi , Ahmedabad and Kolkata seem to top the charts. Ahmedabad has maximum concenterations of NO2,SO2 as well as CO levels.

#Let's also look at the above data visually to get a better perspective

x2= city_day[['PM2.5','City']].groupby(["City"]).median().sort_values(by='PM2.5',ascending=False).reset_index()

x3 = city_day[['PM10','City']].groupby(["City"]).median().sort_values(by='PM10',ascending=False).reset_index()



from plotly.subplots import make_subplots

fig = make_subplots(

    rows=1, cols=2,

    subplot_titles=("PM2.5","PM10"))



fig.add_trace(go.Bar( y=x2['PM2.5'], x=x2["City"],  

                     marker=dict(color=x2['PM2.5'], coloraxis="coloraxis")),

              1, 1)





fig.add_trace(go.Bar( y=x3['PM10'], x=x2["City"],  

                     marker=dict(color=x3['PM10'], coloraxis="coloraxis")),

              1, 2)

fig.update_layout(coloraxis=dict(colorscale='reds'), showlegend=False,plot_bgcolor='white')

fig.update_xaxes(ticks="outside", tickwidth=2,tickangle=45, tickcolor='crimson', ticklen=10,title_text="cities")

fig.update_yaxes(title_text="ug / m3", row=1, col=1)

fig.update_yaxes(title_text="ug / m3", row=1, col=2)

fig.show()
#Effect Of Lockdown on AQI

cities = ['Ahmedabad','Delhi','Bengaluru','Mumbai','Hyderabad','Chennai']



filtered_city_day = city_day[city_day['Date'] >= '2019-01-01']

AQI = filtered_city_day[filtered_city_day.City.isin(cities)][['Date','City','AQI','AQI_Bucket']]

AQI.head()
AQI_pivot = AQI.pivot(index='Date', columns='City', values='AQI')

AQI_pivot.fillna(method='bfill',inplace=True)





from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig = make_subplots(

    rows=6, cols=1,

    #specs=[[{}, {}],

          # [{"colspan": 6}, None]],

    subplot_titles=("Ahmedabad","Bengaluru","Chennai","Delhi",'Hyderabad','Mumbai'))



fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Ahmedabad'],

                    marker=dict(color=AQI_pivot['Ahmedabad'],coloraxis="coloraxis")),

              1, 1)

fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Bengaluru'],

                    marker=dict(color=AQI_pivot['Bengaluru'], coloraxis="coloraxis")),

              2, 1)

fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Chennai'],

                    marker=dict(color=AQI_pivot['Chennai'], coloraxis="coloraxis")),

              3, 1)

fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Delhi'],

                    marker=dict(color=AQI_pivot['Delhi'], coloraxis="coloraxis")),

              4, 1)

fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Hyderabad'],

                    marker=dict(color=AQI_pivot['Hyderabad'], coloraxis="coloraxis")),

              5, 1)

fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Mumbai'],

                    marker=dict(color=AQI_pivot['Mumbai'], coloraxis="coloraxis")),

              6, 1)



fig.update_layout(coloraxis=dict(colorscale='Temps'),showlegend=False,title_text="AQI Levels")



fig.update_layout(plot_bgcolor='white')



fig.update_layout( width=800,height=1200,shapes=[

      dict(

      type= 'line',

      yref= 'paper', y0= 0, y1= 1,

      xref= 'x', x0= '2020-03-25', x1= '2020-03-25'

    )

    ])



fig.show()
#The black vertical line shows the date on which the first phase of lockdown ame into effect in India.

#The above graph shows the variation of various pollutant levels, from Jan 2019 onwards till date.

#Apparantely, all the above Indian cities seem to be a dangerously high level of pollution levels.

#Clearly, there appears to be a rapid decline after 25th March,2020 in all the cities under consideration.