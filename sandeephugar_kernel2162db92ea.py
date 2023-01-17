# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv')
df.dtypes
df.head()
df['Data'] = pd.to_datetime(df['Date'],dayfirst=True)
len(df['Date'].unique())/4
mean['Price'].plot(yerr=var['Price'],ylim=[100000,2500000])
means = df[(df['Type']=='h') & (df['Distance']<13)].sort_values('Date',ascending=False).groupby('Date').mean()

errors = df[(df['Type']=='h') & (df['Distance']<13)].sort_values('Date',ascending=False).groupby('Date').std()
means.columns
means.drop(['Price','Postcode','Lattitude','Longtitude','Distance','BuildingArea','Propertycount','YearBuilt'],axis=1).plot(yerr = errors)
df[df["Type"]=='h'].sort_values('Date',ascending=False).groupby('Date').mean()
sns.kdeplot(df[(df["Suburb"]=="Northcote")

         & (df["Type"]=="u")

         & (df["Rooms"] == 2)]["Price"])
plt.figure(figsize=(20,15))

my_axis = sns.kdeplot(df["Price"][((df["Type"]=="u") &

                                (df["Distance"]>8) &

                                (df["Distance"]<10) &

                                (df["Rooms"] > 2)#&

                                #(df["Price"] < 1000000)

                               )])

my_axis.axis(xmin=0, xmax=2000000)
sns.lmplot('Distance','Price',df[(df['Rooms']<=4) &

                                (df['Rooms']>2) &

                                (df['Type']=='h') &

                                (df['Price']<1000000)].dropna(),hue="Rooms",size=10)
df[(df['Rooms']>2)&(df['Type']=='h')&(df["Landsize"]<5000)][["Landsize","Distance"]].dropna().groupby('Distance').mean().plot()
df.columns
sns.pairplot(df.dropna())
fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(df[df['Type']=='h'].corr(), annot=True)
from sklearn.cross_validation import train_test_split
df_dr = df.dropna().sort_values('Date')
from datetime import date
all_Date = []
df_dr = df_dr
days_since_start = [(x - df_dr["Date"].min()).days for x in df_dr["Date"]]