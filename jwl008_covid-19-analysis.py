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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import folium

import plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected = True)

import plotly.graph_objs as go
df_world = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv',parse_dates = ['last_update'])

print(df_world.info())
features = ['country_region','last_update','confirmed','deaths','report_date_string']

df_world = df_world[features]

print(df_world.head())

country = list(df_world['country_region'].unique())

print(country)
plot_country = ['China','Italy','Russia', 'Iran','India','France','Germany','United Kingdom','Spain','US']

color_list = ['rgba(200,0,0,0.6)', 'rgba(190,110,0,0.6)', 'rgba(170,0,100,0.6)',

              'rgba(0,110,0,0.6)','rgba(150,180,0,0.6)','rgba(0,170,120,0.6)',

              'rgba(0,0,160,0.6)','rgba(110,0,160,0.6)','rgba(0,110,160,0.6)','rgba(100,120,140,0.6)','rgba(100,120,140,0.6)']

data = []

for num,country in enumerate(plot_country):

    df_sub = df_world[df_world.country_region == country].iloc[:120]

    cur = go.Scatter(x = df_sub.last_update, y = df_sub.confirmed, mode = 'lines+markers', name = country ,

                    marker = dict(color = color_list[num]))

    data.append(cur)

layout = dict(title = 'COVID-19 Pandemic', xaxis = dict(title = 'Date'), yaxis = dict(title = 'Confirmed Cases'))

fig = dict(data = data, layout = layout)

iplot(fig)
col_num = 2

row_num = np.ceil(len(plot_country)/2)

fig = plt.figure(figsize= (20,10*row_num))

for i, name in enumerate(plot_country):

    df = df_world[df_world.country_region == name].iloc[:120].copy()

    df[['daily_confirmed','daily_deaths']] = df[['confirmed','deaths']].diff().fillna(0)

    df_0 = df[df.daily_confirmed == df.daily_confirmed.max()]

    date = df_0['last_update'].astype(str).values[0]

    cases = df_0['daily_confirmed'].astype(int).values[0]

    df_1 = df[df.daily_deaths == df.daily_deaths.max()]

    date1 = df_1['last_update'].astype(str).values[0]

    deaths = df_1['daily_deaths'].astype(int).values[0]

    ax = fig.add_subplot(row_num,col_num,i+1)

    ax.set_axisbelow(True)

    plt.tight_layout(pad = 10, w_pad = 5, h_pad = 5)

    plt.plot('last_update','daily_confirmed', data = df,color = 'orange',label = 'Daily Confirmed Cases',ls = '--')

    plt.plot('last_update', 'daily_deaths', data = df, color = 'red', label = 'Daily Deaths Cases',ls = '-.')

    plt.legend(loc = 'best')

    plt.xticks(rotation = 45)

    plt.xlabel('Date')

    plt.ylabel('Number')

    plt.annotate('Peak: '+str(cases) + '\n' + 'Date: '+ (date), xy = (pd.to_datetime(date),cases),

             xytext = (pd.to_datetime(date)-pd.Timedelta(days=15),cases/1.2),

            arrowprops = dict(facecolor = 'gray', alpha = 0.3,shrink = 0.05))

    plt.annotate('Peak: '+str(deaths) + '\n' + 'Date: '+ (date1), xy = (pd.to_datetime(date1),deaths),

             xytext = (pd.to_datetime(date1)-pd.Timedelta(days=15),deaths/0.6),

            arrowprops = dict(facecolor = 'gray', alpha = 0.3,shrink = 0.05))

    plt.title('Daily Confirmed & Deaths Cases in ' + name)

    

    
df_map = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv',parse_dates = ['last_update'])

print(df_map.head())

print(df_map.info())
sizes_c = []

sizes_d = []

other_c = df_map.confirmed.sum()

other_d = df_map.deaths.sum()

for name in plot_country:

    num_c = df_map[df_map.country_region == name]['confirmed'].values[0]

    num_d = df_map[df_map.country_region == name]['deaths'].values[0]

    other_c = other_c - num_c

    other_d = other_d - num_d

    sizes_c.append(num_c)

    sizes_d.append(num_d)

sizes_c.append(other_c)

sizes_d.append(other_d)

labels = plot_country + ['Other']

df_c = pd.DataFrame({'Country':labels,'Confirmed':sizes_c}).sort_values(by=['Confirmed'])

df_d = pd.DataFrame({'Country':labels, 'Deaths':sizes_d}).sort_values(by = ['Deaths'])

cmap = plt.get_cmap('YlOrBr')

color_c = cmap(np.array([10,30,50,70,90,110,130,150,170,190,210]))

cmap = plt.get_cmap('OrRd')

color_d = cmap(np.array([10,30,50,70,90,110,130,150,170,190,210]))

fig = plt.figure(figsize = (20,10))

fig.add_subplot(121)

plt.pie(df_c.Confirmed, explode = (0,0,0,0,0,0,0,0,0,0.1,0), labels = df_c.Country, autopct = '%1.1f%%',shadow = True,colors = color_c)

plt.axis('equal')

plt.title('Confirmed Cases Percentage')

fig.add_subplot(122)

plt.pie(df_d.Deaths, explode = (0,0,0,0,0,0,0,0,0,0,0.1), labels = df_d.Country, autopct = '%1.1f%%',shadow = True,colors = color_d)

plt.axis('equal')

plt.title('Death Cases Percentage')

plt.show()

    

    

df_map = df_map.drop(df_map[df_map.lat.isna()].index)

print(df_map.info())
map_world = folium.Map(location = [10,0],zoom_start = 2, max_zoom = 6,min_zoom = 2)

for i in range(0,186):

    folium.Circle(location = [df_map.iloc[i]['lat'],df_map.iloc[i]['long']],

                  tooltip = "<h5 style = 'text-align:center;font-weight:bold'>" + df_map.iloc[i]['country_region']+

                  "</h5>" + "<hr style='margin:10px;'>"+

                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

        "<li>Confirmed: "+str(df_map.iloc[i]['confirmed'])+"</li>"+

        "<li>Deaths:   "+str(df_map.iloc[i]['deaths'])+"</li>"+

        "<li>Mortality Rate:   "+str(np.round(df_map.iloc[i]['mortality_rate'],2))+"</li>"+

        "</ul>",

                 radius = (int((np.log(df_map.iloc[i]['confirmed']+1.00001)))+0.2)*35000,fill = True,

                  color = 'orange',fill_color = 'yellow').add_to(map_world)

map_world
