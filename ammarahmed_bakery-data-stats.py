# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline
def convert_time(b):
    return b[:2]

# Any results you write to the current directory are saved as output.
path=r'../input/BreadBasket_DMS.csv'
df=pd.read_csv(path)
#we`ll use this for analysis and keep the main dataframe separate, untouched
df_analysis=df.copy()
#Get just the hour of day in the time column
df_analysis['Time']=df['Time'].apply(convert_time)

#check whether it converted successfully.
df_analysis['Time'].head(2)
#now lets find the top 5 most bought items in the bakery
df_most_bought=df_analysis.groupby(['Item'])['Item'].count()
df_most_bought=df_most_bought.sort_values(ascending=False)
df_top5=df_most_bought.head(5)
plt.bar(df_top5.index,df_top5)

#Now lets see what time of the day do we see the most sales
df_grpby=df_analysis.groupby('Time')['Time'].count()

plt.xlabel('hr of the day in 24 hr format')
plt.ylabel('sales made at the hour')
plt.plot(df_grpby.index,
        df_grpby)
#lets see which were the busiest days of the year!
#df_grpby=df_analysis.groupby(['Date','Time'])['Item'].count()
df_grpby=df_analysis.groupby(['Date'])['Item'].count()
df_grpby=df_grpby.sort_values(ascending=False)
df_top5_busiest=df_grpby.head(5)
plt.xlabel('dates')
plt.ylabel('transactions that day')
plt.bar(df_top5_busiest.index,
        df_top5_busiest)

busiest_day = df_analysis['Date']=='2017-02-04'
df_busiest_day=df_analysis[busiest_day].groupby(['Time'])['Item'].count()
plt.xlabel('Hour of the day')
plt.ylabel('transactions made')
plt.plot(df_busiest_day.index,
df_busiest_day)

