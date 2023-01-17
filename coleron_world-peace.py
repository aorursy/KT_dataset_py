# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
regional = pd.read_csv('../input/regional.csv')
regional = pd.read_csv('../input/regional.csv')
data1 = regional['year'].head()
data2 = regional['region'].head()
data3 = regional['christianity_all'].head()
data4 = regional['islam_all'].head()
data5 = regional['buddhism_all'].head()
data6 = regional['judaism_all'].head()

pd.concat([data1,data2,data3,data4,data5,data6],axis =1,ignore_index =False) 
regional = pd.read_csv('../input/regional.csv')
data1 = regional['year'].tail()
data2 = regional['region'].tail()
data3 = regional['christianity_all'].tail()
data4 = regional['islam_all'].tail()
data5 = regional['buddhism_all'].tail()
data6 = regional['judaism_all'].tail()

pd.concat([data1,data2,data3,data4,data5,data6],axis =1,ignore_index =False) 
regional = pd.read_csv('../input/regional.csv')
data1 = regional.head()
data2 = regional.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)
conc_data_row
(regional['region'].unique())
fig, axes = plt.subplots(nrows=1, ncols=2)

CHRIST = regional.groupby(['year','region']).christianity_all.sum()
CHRIST.unstack().plot(kind='area',stacked=False,  colormap= 'jet', grid=True,ax= axes[0],figsize=(11.0,5.0) , legend=False)
axes[0].set_title('Christianity Followers',y=1.0,size=10)
axes[0].set_ylabel('Hundred Millions', y=0.5,size=10)

ISLAM = regional.groupby(['year','region']).islam_all.sum()
ISLAM.unstack().plot(kind='area',stacked=False, colormap= 'jet', grid=True, ax= axes[1], figsize=(11.0,5.0), legend= False,)
axes[1].set_title('Islam Followers',y=1.0,size=10), 
axes[1].set_ylabel('Hundred Millions', y=0.5,size=10)

plt.legend(loc='upper left', title="REGION", numpoints=1, ncol=1, fontsize=10, bbox_to_anchor=(1.1, 0.6))
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2)

BUD = regional.groupby(['year','region']).buddhism_all.sum()
BUD.unstack().plot(kind='area', figsize=(11.0,5.0),stacked=False, legend=False, colormap= 'jet', grid=True, ax= axes[0])
axes[0].set_title('Buddhist Followers',y=1.0,size=10)
axes[0].set_ylabel('Hundred Millions',y=0.5,size=10)

JUD = regional.groupby(['year','region']).judaism_all.sum()
JUD.unstack().plot(kind='area', figsize=(11.0,5.0), stacked=False, legend=False, colormap= 'jet', grid=True, ax= axes[1],)
axes[1].set_title('Judist Followers',y=1.0,size=10)

plt.legend(loc='upper left', title="REGION", numpoints=1, ncol=1, fontsize=10, bbox_to_anchor=(1.1, 0.6))
plt.show()
fig1 = plt.figure(1, figsize=(6, 6))
ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_title("Faiths in 2010 by Region")

labels = 'Christian', 'Islam', 'Buddhist', 'Judist'
fracs = [50.6, 37.3, 11.6, 0.5]
explode = (0.05, 0.05, 0.05, 0.05)
pies = ax.pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%')

plt.show()
regional[regional['region']=='Europe'].plot.area(x='year', grid=True, colormap = 'cubehelix', figsize=(14.0,5.0),
y=['christianity_protestant','christianity_romancatholic','christianity_easternorthodox',
'christianity_anglican', 'christianity_other', 'islam_sunni', 'islam_shi’a', 'islam_alawite', 'islam_other' ])

plt.ylabel('Growth of Followers (Europe)')
plt.legend(loc='upper left', title="CHRISTIAN BRANCH", numpoints=1, ncol=1, fontsize=10, bbox_to_anchor=(1.1, 0.8))
plt.show()
regional[regional['region']=='West. Hem'].plot.area(x='year', grid=True, colormap = 'jet', figsize=(14.0,10.0),
y=['zoroastrianism_all','hinduism_all','sikhism_all', 'shinto_all', 'baha’i_all', 'taoism_all',
'jainism_all', 'confucianism_all', 'animism_all', 'syncretism_all'])

plt.ylabel('Growth of Followers in the Western Hem')
plt.legend(loc='upper left', title="OTHER RELIGIONS", numpoints=1, ncol=1, fontsize=10, bbox_to_anchor=(1.1, 0.8))
plt.show()