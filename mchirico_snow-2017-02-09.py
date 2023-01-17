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
d['date']=d.timeStamp.apply(lambda x: x.date())

d[d.title.str.match(r'.*VEHICLE ACCIDENT.*')].head()

g=d.groupby(['date','twp']).agg(['count'])

g=g.reset_index()

g.columns = g.columns.get_level_values(0)

g.head()
g = g[['date','twp','lat']]

g.columns = ['date','twp','count']

g.head()