# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import fbprophet

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl

df=pd.read_csv("../input/avocado-prices/avocado.csv")

df.head()
##check for null valuea

df.isnull().sum()
##set date as index

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df.head()
df.sort_values(by=['Date'], inplace=True)

#DistPlot

pl.figure(figsize=(12,5))

pl.title("Distribution Price")

ax = sns.distplot(df["AveragePrice"], color = 'b')
#Price of Organic avocados in different cities

organic = sns.factorplot('AveragePrice','region',data=df[df['type']=='organic'],

                   hue='year',

                   size=15,

                   aspect=0.9,

                   palette='magma',

                   join=False,

              )
conventional = sns.factorplot('AveragePrice','region',data=df[df['type']=='conventional'],

                   hue='year',

                   size=15,

                   aspect=0.9,

                   palette='magma',

                   join=False,

              )
df_prophet=df[['Date','AveragePrice']]

df_prophet

#Rename columns

df_prophet=df_prophet.rename(columns={"Date":'ds',"AveragePrice":'y'})

m=fbprophet.Prophet()

m.fit(df_prophet)



#make a dataframe with future dates for one year

future= m.make_future_dataframe(periods=365)

future.tail()
forecast=m.predict(future)



#plotting the prdicted data

m.plot(forecast)
fig1 = m.plot_components(forecast)