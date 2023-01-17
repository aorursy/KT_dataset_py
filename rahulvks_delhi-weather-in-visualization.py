# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

import scipy.stats

import seaborn as sns

import datetime

from pylab import rcParams

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/testset.csv",parse_dates=['datetime_utc'],skipinitialspace=True)
#lenght 

len(data)
data.columns
#'Formatted Date' transformation:

data['Date'] = pd.to_datetime(data['datetime_utc'])

data['year'] = data['Date'].dt.year

data['month'] = data['Date'].dt.month

data['day'] = data['Date'].dt.day

data['hour'] = data['Date'].dt.hour

year_humi = data.groupby(data.year).mean()

pd.stats.moments.ewma(year_humi._hum, 5).plot()

year_humi._hum.plot(linewidth=1)

plt.title('Delh Average Humidity by year')

plt.xlabel('year')
year_heat = data.groupby(data.year).mean()

pd.stats.moments.ewma(year_heat._heatindexm , 5).plot()

year_heat._heatindexm .plot(linewidth=1)

plt.title('Delhi Average Heat by year')

plt.xlabel('year')
year_rain = data.groupby(data.year).mean()

pd.stats.moments.ewma(year_rain._rain, 5).plot()

year_rain._rain.plot(linewidth=1)

plt.title('Delhi Average Rain by year')

plt.xlabel('year')
p = sns.stripplot(data=data, x='year', y='_heatindexm');

p.set(title='Delhi Heat')

dec_ticks = [y if not x%20 else '' for x,y in enumerate(p.get_xticklabels())]

p.set(xticklabels=dec_ticks)
#This code copied and modified, Code Owner Dont Know

#Drawing a heatmap

def facet_heatmap(data, color, **kws):

    values=data.columns.values[3]

    data = data.pivot(index='day', columns='hour', values=values)

    sns.heatmap(data, cmap='coolwarm', **kws)  



#Joining heatmaps of every month in a year 

def weather_calendar(year,weather): 

    dfyear = data[data['year']==year][['month', 'day', 'hour', weather]]

    vmin=dfyear[weather].min()

    vmax=dfyear[weather].max()

    with sns.plotting_context(font_scale=12):

        g = sns.FacetGrid(dfyear,col="month", col_wrap=3) #One heatmap per month

        g = g.map_dataframe(facet_heatmap,vmin=vmin, vmax=vmax)

        g.set_axis_labels('Hour', 'Day')

        plt.subplots_adjust(top=0.9)

        g.fig.suptitle('%s Calendar. Year: %s.' %(weather, year), fontsize=18)

weather_calendar(2006,'_hum')
weather_calendar(2006,'_rain')
weather_calendar(2006,'_fog')