# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

from pylab import *

import scipy.stats                # statistics

from sklearn import preprocessing

import dateutil.parser

from datetime import *







df = pd.read_csv("../input/bikeshare.csv")

# , index_col = 'datetime', parse_dates=True)



print(df.head())

print(df.tail())

print(df.columns.tolist())

print(df.info())

print(df.shape)



#código para obtener las rentas contìnuas

sns.lineplot(x="datetime", y="count", data=df)

# plt.ylim(0, 1500)





nc = (pd.DataFrame(df.groupby('season')['count'].sum()) )

nc.reset_index(inplace=True)

#print(nc)

sns.barplot(x=nc["season"],y=nc["count"], palette=("Blues_d"))
yk = sns.distplot( df["count"])

plt.show()
ax = sns.scatterplot(x="atemp", y="count", data=df)
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
hd = df[df.holiday == 1]

wd = df[df.workingday == 1]

wend = df[(df.holiday == 0) & (df.workingday == 0)]

wend.head()

wd.head()



average(hd['count'])

average(wd['count'])

average(wend['count'])



                     

h = [average(hd['count']),average(wend['count']),average(wd['count'])]

b = ('hd','wd','wend')

    

y = np.arange(len(b))



plt.bar(y,h)

plt.xticks(y,b)

hd['datetime'] = pd.to_datetime(hd['datetime'], errors='coerce')

wd['datetime'] = pd.to_datetime(wd['datetime'], errors='coerce')

wend['datetime'] = pd.to_datetime(wend['datetime'], errors='coerce')



hd['hour'] = hd['datetime'].dt.hour

wd['hour'] = wd['datetime'].dt.hour

wend['hour'] = wend['datetime'].dt.hour



wend.head()

nc4 = pd.DataFrame((hd.groupby('hour')['count'].mean()) )

nc5 = pd.DataFrame((wd.groupby('hour')['count'].mean()) )

nc6 = pd.DataFrame((wend.groupby('hour')['count'].mean()) )

nc4['day_type'] = repeat('hd',len(nc4['count']))

nc5['day_type'] = repeat('wd',len(nc5['count']))

nc6['day_type'] = repeat('wend',len(nc6['count']))



print(nc4)



df4 = pd.concat([nc4,  nc5, nc6])

df4 = df4.reset_index()

df4.head()



#Gráfico de barras promedio rentas por tipo de día. 

g = sns.barplot(x="hour", y="count", hue="day_type", data=df4)

        