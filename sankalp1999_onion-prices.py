# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style



%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/onion-prices-india-2015/daily_retail_price_Onion-upto_apr_2015.csv')
df['Date'] = pd.to_datetime(df.Date)
df.head()
print(df['Centre_Name'].unique())
df_mum = df.loc[df.Centre_Name == 'MUMBAI']
# Lets check if there are missing values. The data is from Government Of India so I hope data is fine.

df_mum.isnull().sum()
style.use('seaborn')

fig = plt.figure(figsize=(20, 10))

plt.scatter(x= df.loc[df.Centre_Name=='DELHI'].Date, y = df.loc[df.Centre_Name=='DELHI'].Price)

plt.gcf().autofmt_xdate()

plt.show()
style.use('fivethirtyeight')

fig = plt.figure(figsize=(20, 10))

plt.scatter(x= df.loc[df.Centre_Name=='MUMBAI'].Date, y = df.loc[df.Centre_Name=='MUMBAI'].Price)

plt.gcf().autofmt_xdate()

plt.show()
df_2012 = df.loc[df.Date.dt.year > 2011]

df_Mumbai = df_2012[df_2012.Centre_Name == 'MUMBAI']
df_Delhi = df_2012[df_2012.Centre_Name == 'DELHI']
style.use('seaborn-bright')

fig = plt.figure(figsize=(20, 10))

plt.plot_date(x= df_Mumbai.Date, y = df_Mumbai.Price, label = 'Mumbai')

plt.plot_date(x = df_Delhi.Date, y = df_Delhi.Price, label = 'Delhi')

plt.gcf().autofmt_xdate()

plt.legend(loc='upper left')

plt.show()