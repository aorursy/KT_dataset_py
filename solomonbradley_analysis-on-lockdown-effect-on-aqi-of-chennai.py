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
df=pd.read_csv(r"/kaggle/input/aqi-of-manalichennai/manali.csv",parse_dates=True)



print(df.columns)

df.columns=['date','pm25','pm10','o3','no2','so2','co']

print(df.columns)

print(df.dtypes)
df["o3"] =pd.to_numeric(df["o3"],errors="coerce")

df["no2"] =pd.to_numeric(df["no2"],errors="coerce")

df["so2"] =pd.to_numeric(df["so2"],errors="coerce")

df['co'] =pd.to_numeric(df["co"],errors="coerce")

df['pm25'] =pd.to_numeric(df["pm25"],errors="coerce")

df['pm10'] =pd.to_numeric(df["pm10"],errors="coerce")



print(df.dtypes)
df.describe()
df.isnull().sum()
print(df.pm10.value_counts())
import missingno as msno
msno.matrix(df)
df.set_index("date")
df.sort_index(axis = 0,ascending=True,inplace=True)
msno.bar(df)
df.columns
df.drop(df.columns[[2,3]],axis=1,inplace=True)
df.columns
plt.figure(figsize=(16,8))

msno.bar(df)
df.fillna(method="ffill",inplace=True)
df.isnull().isnull().sum()
df['date']=pd.to_datetime(df['date'])
df['date'].min()

df['date'].max()
import seaborn as sns
fig, axes = plt.subplots(figsize=(20,6),nrows=4)

df.reset_index().plot(x="date",y="pm25",ax=axes[0],color='coral')

df.reset_index().plot(x="date",y="co",ax=axes[1],color='goldenrod')

df.reset_index().plot (x="date",y="no2",ax=axes[2],color='lightsteelblue')

df.reset_index().plot( x="date",y="so2",ax=axes[3],color='yellowgreen')

df.dtypes
df.set_index('date',inplace=True)

df.index
df['Year'] = df.index.year

df['Month'] = df.index.month

df['Weekday Name'] =df.index.day_name
df["Day"]=df.index.day
df['Weekday Name'] =df.index.day_name()

df['Month Name'] = df.index.month_name()
df.tail(15)

df.loc['2017-11-14']
cols_plot = ['pm25', 'no2', 'so2','co']

axes = df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)

for ax in axes:

    ax.set_ylabel('AQI Reading')
fig, axes = plt.subplots(figsize=(16,25),nrows=4)

sns.barplot(x=df["Month Name"],y=df['so2'],ax=axes[0],color='coral')

sns.barplot(x=df["Month Name"],y=df['pm25'],ax=axes[1],color='yellowgreen').axhline(y=120)

sns.barplot(x=df["Month Name"],y=df['co'],ax=axes[2],color='lightsteelblue').axhline(y=10)

sns.barplot(x=df["Month Name"],y=df['no2'],ax=axes[3],color='goldenrod')


pv1 = pd.pivot_table(df, index=df.index, columns=df.index.year,

                    values='pm25', aggfunc='sum')





pv1.plot(figsize=(25,3))

df.sample(5)
df.dtypes

d2=df.loc[((df.Month >=4) &( df.Month<=6)),["pm25","co","so2","no2","Year","Day"]]
d2.sample(5)
fig, axes = plt.subplots(figsize=(16,25),nrows=4)

sns.lineplot(x=d2.Day, y="co", hue='Year', data=d2,palette=['green','orange','brown','dodgerblue','red',"yellowgreen","black"],ci=None,ax=axes[0])

sns.lineplot(x=d2.Day, y="so2", hue='Year', data=d2,palette=['green','orange','brown','dodgerblue','red',"yellowgreen","black"],ci=None,ax=axes[1])

sns.lineplot(x=d2.Day, y="pm25", hue='Year', data=d2,palette=['green','orange','brown','dodgerblue','red',"yellowgreen","black"],ci=None,ax=axes[2])

sns.lineplot(x=d2.Day, y="no2", hue='Year', data=d2,palette=['green','orange','brown','dodgerblue','red',"yellowgreen","black"],ci=None,ax=axes[3])

pvmp = pd.pivot_table(df, index=df.Year, columns=df["Month"],

                    values='pm25', aggfunc='mean')

print(pvmp)

plt.figure(figsize=(11,9))

sns.heatmap(data=pvmp,fmt=".1f",annot=True,cmap = sns.cm.rocket_r)
plt.figure(figsize=(16,8))

sns.regplot(x=df.index,y=df["pm25"])