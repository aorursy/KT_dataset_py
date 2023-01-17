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



# User added imports

import matplotlib.pyplot as plt

import calendar as cldr

import seaborn as sns
# Read data and separate by ';'

df = pd.read_csv('/kaggle/input/broipr/BrentOilPrices.csv', sep=';')



# Glimpse

df.head()
# Change date format

df['Date'] = pd.to_datetime(df['Date'])



# Glimpse

df.head()
# Add three columns Year, Month and Day from Date

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df['Day'] = df['Date'].dt.day



# Glimpse

df.head()
gp = df.groupby([(df['Year']),(df['Month'])])['Price'].mean()

gp = gp.reset_index()

gp.columns = ['Year', 'Month', 'MeanPrice']



# Map month number to month name

gp['MonthName'] = gp['Month'].apply(lambda x: cldr.month_name[x])



# Merge month and year

gp["MMYY"] = gp["MonthName"].map(str) + ', ' + gp["Year"].map(str)



# Glimpse

gp.head()
# Plot average oil prices from 1987 to 2019

gp.plot(x='MMYY', y='MeanPrice')

plt.xticks(rotation=90)

plt.title('Average oil prices in Brent from May, 1987 to September 2019')

plt.xlabel('')

plt.ylabel('Oil Prices (unit unknown)')
# Plot a kernel density estimation

sns.kdeplot(gp.Year, gp.MeanPrice, cmap="Reds", shade=True, shade_lowest=False)