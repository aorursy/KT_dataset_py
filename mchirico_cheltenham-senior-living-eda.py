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
d=d[(d.timeStamp >= "2016-01-01 00:00:00")]

# Cheltenham

d=d[(d.twp == 'CHELTENHAM')]

# Location

d=d[(d.addr == 'ASHBOURNE RD & PARK AVE')]


d["title"].value_counts()
# Let's look at 8 week intervals



# Why so many falls during one 16W?

p = d.groupby([pd.TimeGrouper('8W'), 'title']).sum().reset_index()

p.columns = ['timeStamp','title','total']  # make columns meaningful

p.head() 




pp = pd.pivot_table(p, values='total', index=['timeStamp'], columns=['title'], aggfunc=np.sum).reset_index()

#pp.columns = pp.columns.get_level_values(0)

pp.head()


# This is puzzeling...why the spike? Warmer weather?

pp[['timeStamp','EMS: FALL VICTIM']]