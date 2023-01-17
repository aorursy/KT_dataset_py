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
# import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.pyplot import figure
import folium
import geopandas as gpd
from folium.plugins import TimestampedGeoJson
import math

from pathlib import Path
data_dir = Path('../input')

import os
os.listdir(data_dir)
#import csv file for all country. We can do comparative analysis later 
cleaned_data = pd.read_csv(data_dir/'corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
cleaned_data.head()
cleaned_data.rename(columns={'ObservationDate': 'date', 
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Last Update':'last_updated',
                     'Confirmed': 'confirmed',
                     'Deaths':'deaths',
                     'Recovered':'recovered'
                    }, inplace=True)

# cases 
cases = ['confirmed', 'deaths', 'recovered', 'active']

# Active Case = confirmed - deaths - recovered
cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths'] - cleaned_data['recovered']

# replacing Mainland china with just China
cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')

# filling missing values 
cleaned_data[['state']] = cleaned_data[['state']].fillna('')
cleaned_data[cases] = cleaned_data[cases].fillna(0)
cleaned_data.rename(columns={'Date':'date'}, inplace=True)

data = cleaned_data
#date stuff
print("External Data")
print(f"Earliest Entry: {data['date'].min()}")
print(f"Last Entry:     {data['date'].max()}")
print(f"Total Days:     {data['date'].max() - data['date'].min()}")

data.head()
#extract rows pertinent to Malaysia only
myr = data[data['country']=='Malaysia']
myr.reset_index(drop=True, inplace=True)
myr
#
figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('Confirmed & Active Cases in Malaysia')
plt.xlabel('Date')
plt.ylabel('# of Cases')

plt.plot(myr.date,myr.confirmed,'-', label='Confirmed')
plt.plot(myr.date,myr.active,'-', label= 'Active Cases')



plt.grid(alpha=0.2)
plt.legend(loc="upper left")
fig=plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('Number of Cases Since RMO')
plt.xlabel('Date')
plt.ylabel('# of Cases')

drmo=myr.date[56:]
crmo=myr.confirmed[56:]
plt.plot(drmo,crmo,color='#3186cc', alpha=0.8)
plt.grid(alpha=0.2)
#number of new cases calculation
n=1
newcase=[0]
while n < len(myr.confirmed):
    newcase.append(myr.confirmed[n] - myr.confirmed[n-1])
    n+=1
    
day = np.array(range(1,len(newcase)+1))
figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')

plt.xlabel('Day')
plt.ylabel('# of New Cases')
plt.title('New Cases in Malaysia Starting from 22-1-2020')


plt.bar(day,newcase, label= 'New Cases', color='#4b8bbe')

plt.legend(loc="upper left")
plt.grid(alpha=0.2)
# Lets normalize our data
day_ =day/max(day)
newcase_ =newcase/max(newcase)
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, day_, newcase_)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
x = day/max(day)
figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
y = sigmoid(x, *popt)
plt.plot(day_, newcase_, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('Cases')
plt.xlabel('Days')
plt.show()
figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Date')
plt.ylabel('# of Cases')
plt.title('Closed Cases in Malaysia')


plt.plot(myr.date,myr.recovered,'-', label='Recovered', color='blue')
plt.plot(myr.date,myr.deaths,'-', label= 'Deaths',color='green')
#plt.plot(myr.date,newcase,'.', label= 'New Cases', color='r')

plt.grid(alpha=0.2)
plt.legend(loc="upper left")
#load data
statepath=Path(data_dir/'covid19-world-malaysiaby-state-dataset/my_bystate.csv')
dfstate=pd.read_csv(statepath,encoding='latin1')
dfstate.head()
#dataframe for each state

ked = dfstate[dfstate['state']=='Kedah']
mel = dfstate[dfstate['state']=='Melaka']
kel = dfstate[dfstate['state']=='Kelantan']
kl = dfstate[dfstate['state']=='KL&Putrajaya']
per = dfstate[dfstate['state']=='Perlis']
pen = dfstate[dfstate['state']=='P.Pinang']
n9 = dfstate[dfstate['state']=='N. Sembilan']
joh = dfstate[dfstate['state']=='Johor']
pah = dfstate[dfstate['state']=='Pahang']
sab = dfstate[dfstate['state']=='Sabah']
sar = dfstate[dfstate['state']=='Sarawak']
lab = dfstate[dfstate['state']=='Labuan']
sel = dfstate[dfstate['state']=='Selangor']
ter = dfstate[dfstate['state']=='Terengganu']
rak = dfstate[dfstate['state']=='Perak']
import matplotlib.ticker as ticker
import matplotlib.animation as animation
current_date= '12-04-20'
dt29 = dfstate[dfstate['date'].eq(current_date)].sort_values(by='total').head(15)
dt29
colors = dict(zip(
    ['Labuan','Perlis','Melaka','Terengganu', 'Kedah', 'Pahang', 'P.Pinang', 'Kelantan', 'Sarawak', 'N.Sembilan', 'Perak', 'Sabah', 'Johor', 'KL&Putrajaya', 'Selangor'],
    ['#adb0ff', '#ffb3ff', '#90d595', '#e48381',
     '#aafbff', '#f7bb5f', '#eafb50','red','brown','blue','yellow','blue','green','purple','gray']
))
group_lk = dfstate['state'].to_dict()
def draw_barchart(date):
    dt29 = dfstate[dfstate['date'].eq(date)].sort_values(by='total').head(15)
    ax.clear()
    ax.barh(dt29['state'],dt29['total'], color=[colors[x] for x in dt29['state']])
    #dx = dff['value'].max() / 200
    
    # iterate over the values to plot labels and values (Tokyo, Asia, 38194.2)
    for i, (total, state) in enumerate(zip(dt29['total'], dt29['state'])):
        ax.text(total, i,     state,            ha='right')  # Tokyo: name
        #ax.text(total, i-.25, group_lk[name],  ha='right')  # Asia: group name
        ax.text(total, i,     total,           ha='left')   # 38194.2: value
        
    # ... polished styles
    ax.text(1, 0.4, date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, 'Cases', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'Number of Cases By State',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    ax.text(1, 0, 'by Khairul Hafiz; credit Khairul Hafiz', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
from IPython.display import HTML
fig, ax = plt.subplots(figsize=(20, 12))
animator = animation.FuncAnimation(fig, draw_barchart, frames= ked.date)
HTML(animator.to_jshtml()) 
figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('States with The Highest Cases in Malaysia')

plt.plot(kl.date,kl.total, label='KL & Putrajaya')
plt.plot(joh.date,joh.total, label='Johor')
plt.plot(sab.date,sab.total, label='Sabah')
plt.plot(sel.date,sel.total, label='Selangor')


plt.grid(alpha=0.2)
plt.legend(loc="upper left")
plt.xlabel('Date')
plt.ylabel('# of Cases')
figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('Other States and Their Respective Cases in Malaysia')

plt.plot(ked.date,ked.total, label='Kedah')
plt.plot(mel.date,mel.total, label='Melaka')
plt.plot(kel.date,kel.total, label='Kelantan')
plt.plot(per.date,per.total, label='Perlis')
plt.plot(pen.date,pen.total, label='Pulau Pinang')
plt.plot(n9.date,n9.total, label='Negeri Sembilan')
plt.plot(pah.date,pah.total, label='Pahang')
plt.plot(lab.date,lab.total, label='Labuan')
plt.plot(ter.date,ter.total, label='Terengganu')
plt.plot(rak.date,rak.total, label='Perak')

plt.grid(alpha=0.2)
plt.legend(loc="upper left")
plt.xlabel('Date')
plt.ylabel('# of Cases')
latest=dfstate[dfstate['date']=='12-04-20']
latest.reset_index(level=0,drop=True, inplace=True)
latest.set_index('state', inplace=True)
latest.sort_values(by='death', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
latest
latest.index
from folium import plugins
from folium.plugins import HeatMap

map_hooray = folium.Map(width=1200,height=800,location=[4.000, 102.295999],
                    zoom_start = 6.5) 

# Ensure you're handing it floats
latest['Latitude'] = latest['Latitude'].astype(float)
latest['Longitude'] = latest['Longitude'].astype(float)

# Filter the DF for rows, then columns, then remove NaNs
#heat_df = df_acc[df_acc['Speed_limit']=='40'] # Reducing data size so it runs faster
#heat_df = df_acc[df_acc['Year']=='2007'] # Reducing data size so it runs faster
heat_df = latest[['Latitude', 'Longitude']]
heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])

# List comprehension to make out list of lists
heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(map_hooray)

# Create weight column, using date
heat_df['Weight'] = ked['date'].str[0:2]
heat_df['Weight'] = heat_df['Weight'].astype(float)
heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude', 'Weight'])

# List comprehension to make out list of lists
heat_data = [[[row['Latitude'],row['Longitude']] for index, row in heat_df[heat_df['Weight'] == i].iterrows()] for i in range(13,25)]

# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(map_hooray)

# Display the map
map_hooray

