"""NOAA Named storm history

    https://www.nhc.noaa.gov/climo/images/AtlanticStormTotalsTable.pdf

    

    sea surface temperature from pre-industrial times



    source :

    US EPA climate change indicators

    https://www.epa.gov/climate-indicators/climate-change-indicators-sea-surface-temperature    

"""



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

NS_FILE = '../input/tropical-cyclone-ace/named_storms_history.csv'

SS_FILE = '../input/tropical-cyclone-ace/sea_surface_temperatures.txt'
ns = pd.read_csv(NS_FILE).dropna()

ss = pd.read_csv(SS_FILE)
ns['h-ratio'] = ns['Hurricanes']/ns['Tropical Storms']

ns['major'] = ns['Major Hurricanes']/ns['Tropical Storms']
ns.tail()
ss.tail()
period = 12 ; base_year = [1854,1878]

ns['year_group'] = ns.apply(lambda x: 

        base_year[0]+period*(math.floor((x['Year']-base_year[0])/period)),axis=1)

ss['year_group'] = ss.apply(lambda x: 

        base_year[1]+period*(math.floor((x['Year']-base_year[1])/period)),axis=1)
ns.tail()
ns_pvt = pd.pivot_table(ns,index=['year_group'],values=['Tropical Storms','Major Hurricanes','major','h-ratio'],aggfunc='mean').reset_index()

ss_pvt = pd.pivot_table(ss,index=['year_group'],values=['Annual anomaly'],aggfunc='mean').reset_index()

df = pd.merge(ns_pvt,ss_pvt,on='year_group')

df.rename(columns={'Annual anomaly':'sea surface temp+'},inplace=True)
df.tail()
df.corr()
import seaborn as sb

sb.barplot(x='sea surface temp+',y='Major Hurricanes',data=df)
import matplotlib.pyplot as plt

sb.set(rc={'figure.figsize':(16, 8)})

f, axes = plt.subplots(2,1,figsize=(16,8))

sb.barplot(x='year_group',y='Major Hurricanes',label='major hurricanes',data=df,ax=axes.flat[0])

sb.barplot(x='year_group',y='sea surface temp+',label='sea temp',data=df,ax=axes.flat[1])