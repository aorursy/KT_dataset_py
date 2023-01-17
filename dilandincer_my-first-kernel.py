# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data4.head(10)

#ilk on veriyi inceleyerek on bilgi sahibi olabiliriz
data1 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

data2 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

data3 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

data4 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

data5 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

data6 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
print(data3.info)

#veri hakkinda genel bir bilgi sahibi olabilriz
data6.columns



#Columnlari inceleyebiliriz
#Line Plot 

#Bu grafikten yararlanarak hastaligin tedavi edilebilirligi acisindan bir cikarimda bulunabiliriz.



data6.Confirmed.plot(kind = 'line' , color = 'g', label = 'Confirmed',linewidth=2,alpha=1,grid=True,linestyle=':')

data6.Deaths.plot(kind = 'line',color= 'r',label= 'Deaths',linewidth = 2,alpha = 1 , grid = True,linestyle= ':')

data6.Recovered.plot(kind = 'line',color = 'b', label='Recovered',linewidth = 2,alpha = 1, grid=True,linestyle = ':')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()

print(data4.info)
# Scatter Plot 

#Bu plotu inceleyerek hastalik ve olum oranlari arasinda bir cikarimda bulunabiliriz

data6.plot(kind='scatter', x='Confirmed', y='Deaths',alpha = 0.5,color = 'red')

plt.xlabel('Confirmed')             

plt.ylabel('Deaths')

plt.title('Confirmed Deaths')           

plt.show()
# Histogram

# Zamanla olum sayisindaki artisi gozlemleyebiliriz

data6.Deaths.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data1.corr()
data2.corr()
data3.corr()
data4.corr()
data5.corr()
data6.corr()
#Correlation Map

#Bu harita ile birlikte vakalari, tedavi edilenleri ve olumle sonuclanan vakalarin birbiriyle olan iliski yakinligini inceleyebiliriz

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data6.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()