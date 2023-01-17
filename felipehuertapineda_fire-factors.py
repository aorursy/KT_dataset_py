#%%

import pandas as pd 

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt
data=pd.read_csv('../input/testing1/forestfires.csv', usecols=['month','day','temp','wind','rain','area'])

data.day.unique()

conversion_month={'mar':'03', 'oct':'10', 'aug':'08', 'sep':'09', 'apr':'04', 'jun':'06', 'jul':'07', 'feb':'02', 'jan':'01','dec':'12', 'may':'05', 'nov':'11'}

data['month']=data['month'].map(lambda x:conversion_month[x])

data['month']=data['month'].map(lambda x:datetime.strptime(x, r'%m'))

del data['day']

grouped_by_month=data.groupby(by='month').mean()

grouped_by_month.plot()

plt.show()