import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')



# Read data 

d=pd.read_csv("../input/911.csv",

    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],

    dtype={'lat':str,'lng':str,'desc':str,'zip':str,

                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 

     parse_dates=['timeStamp'],date_parser=dateparse)





# Set index

d.index = pd.DatetimeIndex(d.timeStamp)

d=d[(d.timeStamp >= "2016-01-01 00:00:00")]
# Note way to create month

datetime.datetime.now().strftime("%m %B")
# Use caps because these will be displayed in graph

d['Month'] = d['timeStamp'].apply(lambda x: x.strftime('%m %B'))

d['Hour'] = d['timeStamp'].apply(lambda x: x.strftime('%H'))

p=pd.pivot_table(d, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)

p.head()
# These are all the calls

ax = sns.heatmap(p,cmap="YlGnBu")

ax.set_title('All 911 Calls');
# Vehicle Accident -- yes, there is FIRE; maybe we should have include?

# Put this in a variable 'g'

g = d[(d.title.str.match(r'EMS:.*VEHICLE ACCIDENT.*') | d.title.str.match(r'Traffic:.*VEHICLE ACCIDENT.*'))]

g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))

g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))

p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)

p.head()
ax = sns.heatmap(p,cmap="YlGnBu")

ax.set_title('Vehicle  Accidents - All Townships ');
# Take a look at one TWP

# Copy all the code -- makes it easier to try different things

# 

g = d[(d.title.str.match(r'EMS:.*VEHICLE ACCIDENT.*') | d.title.str.match(r'Traffic:.*VEHICLE ACCIDENT.*'))]

g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))

g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))

g = g[g['twp']== 'CHELTENHAM']

p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)

p.head()

ax = sns.heatmap(p,cmap="YlGnBu")

ax.set_title('Vehicle  Accidents - Cheltenham ');