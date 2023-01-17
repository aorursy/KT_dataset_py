# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.chdir('../input/')
df=pd.read_csv('hotel-booking-demand/hotel_bookings.csv')

df.head()
df.describe()
df.columns
df.info()
df.isna().sum()
fig,axes = plt.subplots(1,1,figsize=(15,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.drop('company',axis=1,inplace=True)
df
fig,axes = plt.subplots(1,1,figsize=(10,7))

sns.heatmap(df.corr())

plt.show()
cols = df.columns

for i in cols:

    print('\n',i,'\n',df[i].unique(),'\n','-'*80)
def time_conv(x):

    return datetime.datetime.strptime(x,'%Y-%m-%d').date()



df['reservation_status_date'] = df['reservation_status_date'].apply(lambda x : time_conv(x))
(df.reservation_status_date.apply(lambda x : x.year)).value_counts().plot(kind='bar')

plt.show()
sns.barplot(x="hotel", y="is_canceled", data=df.groupby("hotel").is_canceled.count().reset_index())

plt.ylabel ('count')

plt.show()
sns.barplot(x="hotel", y="lead_time", hue="is_canceled", data=df.groupby(["hotel","is_canceled"]).lead_time.count().reset_index())

plt.ylabel('count')

plt.show()
sns.countplot(x="market_segment", data = df,palette = 'terrain',order = df['market_segment'].value_counts().index)

plt.xticks(rotation = 90)

plt.show()
sns.countplot(x="reservation_status", data = df, palette="terrain")

plt.show()
sns.countplot(x="deposit_type", data = df, palette="spring",order = df['deposit_type'].value_counts().index)

plt.show()