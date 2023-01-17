import pandas as pd

import numpy as np



data = pd.read_csv('../input/database.csv')

data.sample(10)
import calendar

import time

import datetime



month = {v: k for k,v in enumerate(calendar.month_name)}



def dateToTimestamp(d):

    t = d.split()

    d = '{0}/{1}/{2}'.format(month[t[1]], t[0], t[2])

    return time.mktime(datetime.datetime.strptime(d, "%m/%d/%Y").timetuple())



data['timestamp'] = data['Launch Date'].apply(dateToTimestamp)
import seaborn as sns

import matplotlib

%matplotlib inline



sns.set_context("notebook")

sns.set_style("whitegrid")



def myFormatter(x, pos):

    return datetime.datetime.fromtimestamp(x).strftime('%Y-%m')



def plotOverTime(col):

    ax = sns.swarmplot(x="timestamp", y=col, data=data)

    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(myFormatter))

    ax.set(xlabel='Date')
plotOverTime('Launch Site')
plotOverTime('Customer Name')
plotOverTime('Vehicle Type')
plotOverTime('Payload Orbit')
plotOverTime('Customer Type')
plotOverTime('Customer Country')
plotOverTime('Mission Outcome')
plotOverTime('Landing Type')