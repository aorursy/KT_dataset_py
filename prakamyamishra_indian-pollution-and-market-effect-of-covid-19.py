# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# !pip3 install geopandas
# !pip3 install descartes
# !pip3 install shapely
# !pip3 install dash
# !pip3 install contextily
# !pip3 install geopy
# !pip3 install mercantile
# !pip3 install pillow
# !pip3 install rasterio
# !pip3 install joblib
# !pip3 install osmnx
import geopandas as gpd
import matplotlib
import seaborn as sns
import geopandas as gpd
# !pip3 install yfinance
# !pip3 install lxml
# !pip3 install html5lib
# !pip3 install bsedata
# !pip3 install pandas_datareader
import pandas_datareader
import pandas as pd
from pandas_datareader import data
!pip install mplfinance
import  mplfinance as mpf
import matplotlib.dates as mdates
geo_location = pd.read_csv("/kaggle/input/india-aq-dataset-2020/India AQ 2020/Geo-Locations.csv")
geo_location = geo_location[:-1]
geo_location.head()
aq = pd.read_csv("/kaggle/input/india-aq-dataset-2020/India AQ 2020/India-Full.csv",error_bad_lines=False)
aq_updated = aq[['city','local','value','latitude','longitude']]
aq_updated.head()
date_time = []
for index, row in aq_updated.iterrows():
    date_time.append(" ".join(row['local'].split("+")[0].split("T")))
aq_updated['time'] = date_time
aq_updated.head()
print(max(aq_updated['latitude']))
print(min(aq_updated['latitude']))
print(max(aq_updated['longitude']))
print(min(aq_updated['longitude']))
unique_values =  pd.unique(aq_updated['time'])
print(len(unique_values))
for i,v in enumerate(unique_values):
    d = aq_updated[aq_updated['time']==v]
    print("Index "+str(len(unique_values)-i)+": "+str(len(d)))
date = []
for index, row in aq_updated.iterrows():
    date.append(row['time'].split(" ")[0])
aq_updated['date'] = date
aq_updated.head()
data_reverse = aq_updated.reindex(index=aq_updated.index[::-1])
data_reverse.head()
delhi = data_reverse[data_reverse['city']=='Delhi']
mumbai = data_reverse[data_reverse['city']=='Mumbai']
Kanpur = data_reverse[data_reverse['city']=='Kanpur']
Faridabad = data_reverse[data_reverse['city']=='Faridabad']
Varanasi = data_reverse[data_reverse['city']=='Varanasi']
Patna = data_reverse[data_reverse['city']=='Patna']
Lucknow = data_reverse[data_reverse['city']=='Lucknow']
Agra = data_reverse[data_reverse['city']=='Agra']
frames = [delhi, mumbai, Kanpur, Faridabad,Varanasi,Patna]
result = pd.concat(frames)

print(len(delhi))
print(len(mumbai))

fig, ax = plt.subplots(nrows=8, sharex=True, figsize=(40,20))
matplotlib.rcParams.update({'font.size': 50})
delhi.plot(kind='line',x='date',y='value',ax=ax[0], label='Delhi')
mumbai.plot(kind='line',x='date',y='value',ax=ax[1], label='Mumbai')
Kanpur.plot(kind='line',x='date',y='value',ax=ax[2], label='Kanpur')
Faridabad.plot(kind='line',x='date',y='value',ax=ax[3], label='Faridabad')
Varanasi.plot(kind='line',x='date',y='value',ax=ax[4], label='Varanasi')
Patna.plot(kind='line',x='date',y='value',ax=ax[5], label='Patna')
Lucknow.plot(kind='line',x='date',y='value',ax=ax[6], label='Lucknow')
Agra.plot(kind='line',x='date',y='value',ax=ax[7], label='Agra')

plt.xticks(rotation=90)
plt.show()
result.head()
sns.set(style="ticks", context="talk",font_scale=3.5,rc={'figure.figsize':(50, 30)})
plt.style.use("dark_background")

# fig, ax = plt.subplots()
# fig.set_size_inches(50, 20)
g = sns.relplot(x="date", y="value",markers=True,style="city", hue="city", height=20, aspect=50/20, kind="line",ci="sd",legend = 'full',data=result)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i%10 == 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels, rotation=90) # set new labels
    ax.axhline(35.4, ls='--',color='r',linewidth=3)
g.add_legend(bbox_to_anchor=(1.05, 0), borderaxespad=0.1)
plt.title('Time series plot of PM2.5 values of the top 6 polluted places of India')

plt.savefig("compare.png")
plt.show()

geo_location = pd.read_csv("/kaggle/input/india-aq-dataset-2020/India AQ 2020/Geo-Locations.csv")
geo_location = geo_location[:-1]
geo_location.head()

aq_others_delhi = pd.read_csv("/kaggle/input/india-aq-dataset-2020/India AQ 2020/all_delhi.csv",error_bad_lines=False)
aq_others_faridabad = pd.read_csv("/kaggle/input/india-aq-dataset-2020/India AQ 2020/all_faridabad.csv",error_bad_lines=False)
aq_others_kanpur = pd.read_csv("/kaggle/input/india-aq-dataset-2020/India AQ 2020/all_kanpur.csv",error_bad_lines=False)
aq_others_mumbai = pd.read_csv("/kaggle/input/india-aq-dataset-2020/India AQ 2020/all_mumbai.csv",error_bad_lines=False)
aq_others_patna = pd.read_csv("/kaggle/input/india-aq-dataset-2020/India AQ 2020/all_patna.csv",error_bad_lines=False)
aq_others_varanasi = pd.read_csv("/kaggle/input/india-aq-dataset-2020/India AQ 2020/all_varanasi.csv",error_bad_lines=False)
frames = [
    aq_others_delhi, 
    aq_others_faridabad, 
    aq_others_kanpur, 
    aq_others_mumbai,
    aq_others_patna,
    aq_others_varanasi
]
aq_others = pd.concat(frames)
date_time = []
for index, row in aq_others.iterrows():
    date_time.append(" ".join(row['local'].split("+")[0].split("T")))
aq_others['time'] = date_time
date = []
for index, row in aq_others.iterrows():
    date.append(row['time'].split(" ")[0])
aq_others['date'] = date
aq_others.head()
uni_chemicals = pd.unique(aq_others['parameter'])
print(uni_chemicals)
uni_chemicals = pd.unique(aq_others['parameter'])
cities = ['Delhi', 'Mumbai', 'Kanpur', 'Faridabad','Varanasi','Patna']
#     if chem == 'no2':
#         p='YlOrRd'
#     elif chem == 'co':
#         p='YlGnBu'
#     elif chem == 'pm10':
#         p = 'RdBu'
#     elif chem == 'o3':
#         p = 'PiYG'
#     elif chem == 'so2':
#         p = 'Accent_r' 
d = aq_others[aq_others['parameter']=='no2']
sns.set(style="ticks", context="talk",font_scale=3.5,rc={'figure.figsize':(50, 30)})
plt.style.use("dark_background")
g = sns.relplot(x="date", y="value", palette='YlOrRd',markers=True,style="city", hue="city", height=20, aspect=50/20, kind="line",ci="sd",legend = 'full',data=d)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i%10 == 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels, rotation=90) # set new labels
plt.title('Pollutant levels plot for no2')
g.add_legend(bbox_to_anchor=(1.05, 0), borderaxespad=0.1)
# plt.tight_layout()
plt.ylim(0, 250)
plt.savefig("no2.png")
plt.show()

d = aq_others[aq_others['parameter']=='co']
sns.set(style="ticks", context="talk",font_scale=3.5,rc={'figure.figsize':(50, 30)})
plt.style.use("dark_background")
g = sns.relplot(x="date", y="value", palette='YlGnBu',markers=True,style="city", hue="city", height=20, aspect=50/20, kind="line",ci="sd",legend = 'full',data=d)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i%10 == 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels, rotation=90) # set new labels
plt.title('Pollutant levels plot for co')
g.add_legend(bbox_to_anchor=(1.05, 0), borderaxespad=0.1)
# plt.tight_layout()
plt.ylim(0, 4500)
plt.savefig("co.png")
plt.show()

d = aq_others[aq_others['parameter']=='pm10']
sns.set(style="ticks", context="talk",font_scale=3.5,rc={'figure.figsize':(50, 30)})
plt.style.use("dark_background")
g = sns.relplot(x="date", y="value", palette='PuRd',markers=True,style="city", hue="city", height=20, aspect=50/20, kind="line",ci="sd",legend = 'full',data=d)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i%10 == 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels, rotation=90) # set new labels
plt.title('Pollutant levels plot for pm10')
g.add_legend(bbox_to_anchor=(1.05, 0), borderaxespad=0.1)
# plt.tight_layout()
plt.ylim(0, 400)
plt.savefig("pm10.png")
plt.show()

d = aq_others[aq_others['parameter']=='o3']
sns.set(style="ticks", context="talk",font_scale=3.5,rc={'figure.figsize':(50, 30)})
plt.style.use("dark_background")
g = sns.relplot(x="date", y="value", palette='PiYG',markers=True,style="city", hue="city", height=20, aspect=50/20, kind="line",ci="sd",legend = 'full',data=d)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i%10 == 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels, rotation=90) # set new labels
plt.title('Pollutant levels plot for o3')
g.add_legend(bbox_to_anchor=(1.05, 0), borderaxespad=0.1)
# plt.tight_layout()
plt.ylim(0, 400)
plt.savefig("o3.png")
plt.show()

d = aq_others[aq_others['parameter']=='so2']
sns.set(style="ticks", context="talk",font_scale=3.5,rc={'figure.figsize':(50, 30)})
plt.style.use("dark_background")
g = sns.relplot(x="date", y="value", palette='Accent_r',markers=True,style="city", hue="city", height=20, aspect=50/20, kind="line",ci="sd",legend = 'full',data=d)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        if(i%10 == 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels, rotation=90) # set new labels
plt.title('Pollutant levels plot for so2')
g.add_legend(bbox_to_anchor=(1.05, 0), borderaxespad=0.1)
# plt.tight_layout()
plt.ylim(0, 250)
plt.savefig("so2.png")
plt.show()


start_date = '2020-02-01'
end_date = '2020-04-20'
# Set the ticker
ticker = 'TATAMOTORS.NS'
data_TATAMOTORS = data.get_data_yahoo(ticker, start_date, end_date)
date = data_TATAMOTORS.index
data_TATAMOTORS['date'] = date
ticker = 'LT.NS'
data_LT = data.get_data_yahoo(ticker, start_date, end_date)
date = data_LT.index
data_LT['date'] = date
ticker = 'IRCTC.NS'
data_IRCTC = data.get_data_yahoo(ticker, start_date, end_date)
date = data_IRCTC.index
data_IRCTC['date'] = date
ticker = 'INDIGO.NS'
data_INDIGO = data.get_data_yahoo(ticker, start_date, end_date)
date = data_INDIGO.index
data_INDIGO['date'] = date
ticker = 'DMART.NS'
data_DMART = data.get_data_yahoo(ticker, start_date, end_date)
date = data_DMART.index
data_DMART['date'] = date
data_LT = data_LT[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
data_TATAMOTORS = data_TATAMOTORS[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
data_IRCTC = data_IRCTC[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
data_INDIGO = data_INDIGO[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
data_DMART = data_DMART[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# data_DMART["date"] = data_DMART["date"].apply(mdates.date2num)
plt.rc('font', size=20)          
plt.rc('axes', titlesize=20)    
plt.rc('axes', labelsize=20) 
mpf.plot(
    data_LT,
    type='candle',
    mav=3,
    volume=True,
    figscale=2.5,
    title='L&T: 1 February 2020 - 20 April 2020',
    ylabel='Candles',
    ylabel_lower='Volume',
    style='nightclouds'
)
plt.rc('font', size=20)          
plt.rc('axes', titlesize=20)    
plt.rc('axes', labelsize=20) 
mpf.plot(
    data_TATAMOTORS,
    type='candle',
    mav=3,
    volume=True,
    figscale=2.5,
    title='TATAMOTORS: 1 February 2020 - 20 April 2020',
    ylabel='Candles',
    ylabel_lower='Volume',
    style='nightclouds'
)

plt.rc('font', size=20)          
plt.rc('axes', titlesize=20)    
plt.rc('axes', labelsize=20) 
mpf.plot(
    data_IRCTC,
    type='candle',
    mav=3,
    volume=True,
    figscale=2.5,
    title='IRCTC: 1 February 2020 - 20 April 2020',
    ylabel='Candles',
    ylabel_lower='Volume',
    style='nightclouds'
)

plt.rc('font', size=20)          
plt.rc('axes', titlesize=20)    
plt.rc('axes', labelsize=20) 
mpf.plot(
    data_INDIGO,
    type='candle',
    mav=3,
    volume=True,
    figscale=2.5,
    title='INDIGO: 1 February 2020 - 20 April 2020',
    ylabel='Candles',
    ylabel_lower='Volume',
    style='nightclouds'
)

plt.rc('font', size=20)          
plt.rc('axes', titlesize=20)    
plt.rc('axes', labelsize=20) 
mpf.plot(
    data_DMART,
    type='candle',
    mav=3,
    volume=True,
    figscale=2.5,
    title='DMART: 1 February 2020 - 20 April 2020',
    ylabel='Candles',
    ylabel_lower='Volume',
    style='nightclouds'
)
