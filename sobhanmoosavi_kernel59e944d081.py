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
weather_events = pd.read_csv('/kaggle/input/us-weather-events/US_WeatherEvents_2016-2019.csv')

weather_events.head()
# this part is adopted from https://www.kaggle.com/biphili/road-accidents-in-us



import matplotlib.pyplot as plt

import seaborn as sns

import datetime

plt.style.use('fivethirtyeight')



fig,ax=plt.subplots(1,2,figsize=(15,8))

clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')

weather_events.State.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])

ax[0].set_title("Top 10 Weather Event Prone States",size=20)

ax[0].set_xlabel('States',size=18)





count=weather_events['State'].value_counts()

groups=list(weather_events['State'].value_counts().index)[:10]

counts=list(count[:10])

counts.append(count.agg(sum)-count[:10].agg('sum'))

groups.append('Other')

type_dict=pd.DataFrame({"group":groups,"counts":counts})

clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')

qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])

plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 

plt.subplots_adjust(wspace =0.5, hspace =0)

plt.ioff()

plt.ylabel('')
f,ax=plt.subplots(1,1,figsize=(10,8))

sns.countplot('Type',data=weather_events,ax=ax,order=weather_events['Type'].value_counts().index)

ax.set_title('Event Type Distribution')

plt.show()