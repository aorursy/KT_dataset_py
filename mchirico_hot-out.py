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


k=d[(d.title == 'EMS: DEHYDRATION')]

k.head()
# Note way to create month

datetime.datetime.now().strftime("%m %B")



# Use caps because these will be displayed in graph

d['Month'] = d['timeStamp'].apply(lambda x: x.strftime('%m %B'))

d['Hour'] = d['timeStamp'].apply(lambda x: x.strftime('%H'))



k=d[(d.title == 'EMS: DEHYDRATION')]

p=pd.pivot_table(k, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)

p.head()
ax = sns.heatmap(p,cmap="YlGnBu")

ax.set_title('EMS: DEHYDRATION');