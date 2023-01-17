# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data1 = pd.read_csv('../input/accidents_2017.csv')
data2 = pd.read_csv('../input/transports.csv')
data1.info()  #accidents
data2.info() #transports
data1.corr()
data2.corr()
#correlation map accidents
f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data1.corr(),annot = True,linewidths = .5,fmt = '.1f',ax=ax,center = 0,vmin = -0.3,vmax = 1)
plt.show()
#correlation map transports
f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(data2.corr(),annot = True,linewidths = .5,fmt = '.1f',ax=ax,vmin=0, vmax=1)
plt.show()
data1.head(10)
data1.tail(5)
data2.head(10)
data2.tail(5)
data1.columns
data2.columns

data1.describe()
data1.columns = [each.split()[0]+"_"+each.split()[1] if len(each.split()) >1 else each for each in data1.columns]
data1.columns
# Data1(accidents) visualization
data1.Mild_injuries.plot(kind = 'line', color = 'b',label = 'Mild_injuries',linewidth=1,alpha = 0.7,grid = True,linestyle = '-',figsize = (12,12))
plt.legend(loc='upper right')
plt.xlabel('Sample')              # label = name of label
plt.ylabel('İnjuries')
plt.show()
data1.Serious_injuries.plot(kind = 'line', color = 'r',label = 'Serious_injuries',linewidth=1,alpha = 0.7,grid = True,linestyle = '-',figsize = (12,12))
plt.legend(loc='upper right')
plt.xlabel('Sample')              # label = name of label
plt.ylabel('İnjuries')
plt.show()
data1.plot(kind='scatter', x='Mild_injuries', y='Serious_injuries',alpha = 0.5,color = 'black',figsize=(12,12))
plt.xlabel('Mild-İnjuries')              # label = name of label
plt.ylabel('Serious-İnjuries')
plt.title('Mild-Serious İnjuries') # title = title of plot
plt.show()
data1.Mild_injuries.plot(kind = 'hist',color = 'green',bins = 50,figsize=(12,12))
plt.show()
data1.Serious_injuries.plot(kind = 'hist',color = 'orange',bins = 50,figsize=(12,12))
plt.show()
data2.columns

underground = data2['Transport'] == 'Underground'
len(data2[underground])
tram = data2['Transport'] == 'Tram'
len(data2[tram])
railway = data2['Transport'] == 'Railway (FGC)'
len(data2[railway])
maritime_station = data2['Transport'] == 'Maritime station'
len(data2[maritime_station])
airport_train = data2['Transport'] == 'Airport train'
len(data2[airport_train])
renfe = data2['Transport'] == 'RENFE'
len(data2[airport_train])
cableway = data2['Transport'] == 'Cableway'
len(data2[cableway])
funicular = data2['Transport'] == 'Funicular'
len(data2[funicular])
dictionary = {"Underground":463,
             "tram":65,
             "railway":54,
             "maritime_station":16,
             "airport_train":9,
             "renfe":9,
             "cableway":5,
             "funicular":6}

series = pd.Series(dictionary)
series
data2.columns
# Data2(transports) visualization
data2.plot(kind='scatter', x='Latitude', y='Longitude',alpha = 0.7,color = 'orange',figsize=(12,12))
plt.xlabel('Latitude')              # label = name of label
plt.ylabel('Longitude')
plt.title('Latitude-Longitude') # title = title of plot
plt.show()