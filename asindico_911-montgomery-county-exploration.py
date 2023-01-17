# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: htt in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/911.csv')

df.head()
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(ncols=2, nrows=1,figsize=(12,10))



axarr[0].scatter(df['lng'],df['lat'], s=0.1, alpha=0.4)

axarr[0].set_ylim(39.8, 40.6)

axarr[0].set_xlim(-75.8,-74.9)



day_zero= df[df['timeStamp'].dt.dayofyear==20]

axarr[1].scatter(day_zero['lng'],day_zero['lat'], s=3, alpha=1)

axarr[1].set_ylim(39.8, 40.6)

axarr[1].set_xlim(-75.8,-74.9)

plt.show()
fire=df[df.title.str.contains('Fire')]

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

plt.ylim(39.8, 40.6)

plt.xlim(-75.8,-74.9)

ax.scatter(fire['lng'],fire['lat'], s=0.1, alpha=0.4,color='red')

plt.show()
import seaborn as sns

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

fire_hist = fire.groupby('title')['e'].count().sort_values(ascending=False)

bp = sns.barplot(x=fire_hist[0:20].index,y = fire_hist[0:20].values)

plt.xticks(rotation=90)

plt.show()
ems=df[df.title.str.contains('EMS')]

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

plt.ylim(39.8, 40.6)

plt.xlim(-75.8,-74.9)

ax.scatter(ems['lng'],ems['lat'], s=0.1, alpha=0.4,color='purple')

plt.show()
import seaborn as sns

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

hist = ems.groupby('title')['e'].count().sort_values(ascending=False)

bp = sns.barplot(x=hist[0:20].index,y = hist[0:20].values)

plt.xticks(rotation=90)

plt.show()
traffic=df[df.title.str.contains('Traffic')]

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

plt.ylim(39.8, 40.6)

plt.xlim(-75.8,-74.9)

ax.scatter(traffic['lng'],traffic['lat'], s=0.1, alpha=1,color='gray')

plt.show()
import seaborn as sns

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))

hist = traffic.groupby('title')['e'].count().sort_values(ascending=False)

bp = sns.barplot(x=hist[0:20].index,y = hist[0:20].values)

plt.xticks(rotation=90)

plt.show()