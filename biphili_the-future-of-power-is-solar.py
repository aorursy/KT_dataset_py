# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
#df1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")

df1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv',parse_dates=['DATE_TIME'],index_col=0)

df2 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
df1.tail()
df2.head()
#df1.info()
df1.shape
df1.dtypes
print('Rows     :',df1.shape[0])

print('Columns  :',df1.shape[1])

print('\nFeatures :\n     :',df1.columns.tolist())

print('\nMissing values    :',df1.isnull().values.sum())

print('\nUnique values :  \n',df1.nunique())

def add_feature(df1):

    df1['DATE_TIME']=df1.index.year

    df1['month']=df1.index.month

    df1['day']=df1.index.day

    df1['dayofweek']=df1.index.dayofweek

    df1['hour']=df1.index.hour



add_feature(df1)
df1.head()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1,)

df1.groupby('hour')['TOTAL_YIELD'].sum().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)

plt.xlabel('Hour',fontsize=15)

plt.ylabel('Total Yeild',fontsize=15)

plt.title('Total Yeild on Hourly Basis',fontsize=15)

#ax.tick_params(labelsize=20)

#ax.set_xticklabels(('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

#plt.show()



plt.subplot(2,2,2,)

ax=df1.groupby('dayofweek')['TOTAL_YIELD'].sum().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)

plt.xlabel('Day Of Week',fontsize=15)

plt.ylabel('Total Yeild',fontsize=15)

plt.title('Total Yeild on Weekly Basis',fontsize=15)

ax.set_xticklabels(('Mon','Tue','Wed','Thu','Fri','Sat','Sun'))



plt.subplot(2,2,3,)

df1.groupby('day')['TOTAL_YIELD'].sum().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)

plt.xlabel('Day Of Month',fontsize=15)

plt.ylabel('Total Yeild',fontsize=15)

plt.title('Total Yeild on Weekly Basis',fontsize=15)

#ax.set_xticklabels(('Mon','Tue','Wed','Thu','Fri','Sat','Sun'))



plt.subplot(2,2,4,)

ax1=df1.groupby('month')['TOTAL_YIELD'].sum().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)

plt.xlabel('Month',fontsize=15)

plt.ylabel('Total Yeild',fontsize=15)

plt.title('Total Yeild on Monthly Basis',fontsize=15)

ax1.set_xticklabels(('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))





plt.tight_layout()

pass