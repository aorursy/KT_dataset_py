import matplotlib.pyplot as plt

import plotly.plotly as py

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline

plt.rcParams['figure.figsize']=(12,5)

state_data = '../input/State_time_series.csv'

df=pd.read_csv(state_data)
df.RegionName.groupby(df.RegionName).count().sort_values()[::-1].plot(kind="bar")
df['date'] = pd.to_datetime(df['Date'])

ax=df.plot(x='date', y=['MedianRentalPrice_1Bedroom'])

ax.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2018-01-01'))
ax=df.plot(x='date', y=['MedianRentalPrice_4Bedroom'])

ax.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2018-01-01'))
ax=df.plot(x='date', y=['MedianRentalPrice_5BedroomOrMore'])

ax.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2018-01-01'))
Analysis= ['MedianRentalPrice_1Bedroom',

 'MedianRentalPrice_2Bedroom',

 'MedianRentalPrice_3Bedroom',

 'MedianRentalPrice_4Bedroom']

xaxis=["1bed", "2bed","3bed","4bed"]

y=[]

colors=["blue","black","grey"]

for region in list(df.RegionName.unique())[:3]:

    y=[]

    for x in Analysis:

        ndf= df["RegionName"]==region

        y.append(df[ndf][x].mean())

    plt.bar(xaxis, y, width=0.5, color=colors[random.randint(0,len(colors)-1)])

    print (region)

    plt.show()

x=df.MedianListingPrice_3Bedroom

y=df.MedianListingPrice_4Bedroom

plt.scatter(x,y)
x=df.MedianListingPrice_2Bedroom

y=df.MedianListingPrice_3Bedroom

plt.scatter(x,y)
x=df.MedianListingPrice_1Bedroom

y=df.MedianListingPrice_2Bedroom

plt.scatter(x,y)
x=df.MedianRentalPrice_1Bedroom

y=df.MedianRentalPrice_4Bedroom

plt.scatter(x,y)
x=df.MedianRentalPrice_1Bedroom

y=df.MedianRentalPrice_2Bedroom

plt.scatter(x,y)