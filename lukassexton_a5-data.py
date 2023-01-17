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

import datetime as dt

import math

import statistics

import numpy as np

import scipy.stats

import pandas as pd

!pip install sanpy

!pip install --upgrade sanpy

!pip install sanpy[extras]
import san

san.ApiConfig.api_key = 'f4vxqizm4zcnrgpb_jmuzylo545b5flv6'
import san

san.get("projects/all")
san.available_metrics()
san.get("projects/all")
#Practice Area to learn about Matplot lib using the dataframe ETH 



eth_df = san.get(

    "prices/ethereum",

    from_date="2020-05-04",

    to_date="2020-05-18",

    interval="1d"

    )

print(eth_df)
eth_df.plot(y='volume')
eth_df[['marketcap', 'volume']].plot()
plt.style.available
plt.style.use('fivethirtyeight')

eth_df[['marketcap', 'volume']].plot()
plt.style.use('ggplot') 

eth_df[['marketcap','volume']].plot()
#First Halvening  +-7days 11/28/2012

one_df = san.get(

    "prices/bitcoin",

    from_date="2012-11-21",

    to_date="2012-11-28",

    interval="1d"

)



print(one_df)

#API Turns out the API does not reach back 8 years. 
#Second Halvening  +-7days 07/09/2016

two_df = san.get(

    "prices/bitcoin",

    from_date="2016-07-02",

    to_date="2016-07-15",

    interval="1d"

)

print(two_df)

two_df.plot()

#two_df.mean(axis = 0)

two_df.marketcap.mean(axis = 0)  #It returns the mean of marketcap the columns



two_df.priceUsd.mean(axis = 0)  #It returns the mean of USD the columns



mean = two_df['priceUsd'].mean() # Alternative way to find mean 

two_df[['priceUsd']].plot() # Charts the price of coin over the course time
#Third Halvening  +-7days 05/11/2020

three_df = san.get(

    "prices/bitcoin",

    from_date="2020-05-04",

    to_date="2020-05-18",

    interval="1d"

)

print(three_df)

three_df.plot()

three_df.mean(axis = 0)





three_df.describe() #It returns the summary statistics for the numerical columns

three_df.mean()  #It returns the mean of all the columns

three_df.corr() #It returns the correlation between the columns in the dataframe

three_df.count() #It returns the count of all the non-null values in each dataframe column

three_df.max() #It returns the highest value from each of the columns

three_df.min() #It returns the lowest value from each of the columns

three_df.median() #It returns the median from each of the columns

three_df.std() #It returns the standard deviation from each of the columns

plt.style.use('ggplot') 

three_df[['marketcap','volume']].plot()
three_df[['priceUsd']].plot()



def rank_performance(temp_price):

    if temp_price <= 8900:

        return 'Poor'

    elif temp_price > 8900 and temp_price <=9300:

        return 'Good'

    elif temp_price > 9300:

        return "Excellent"

    



        
def rank_performance(temp_price):

    if temp_price <= 8900:

        return 'Poor'

    elif temp_price > 8900 and temp_price <=9300:

        return 'Good'

    elif temp_price > 9300:

        return "Excellent"
three_df['priceUsd'].apply(rank_performance)
three_df['priceUsd'].apply(rank_performance).value_counts().plot(kind='bar') #Vertical is default

three_df['priceUsd'].apply(rank_performance).value_counts().plot(kind='barh') #Horizontal
# In Class Example 



N = 50

x = np.random.rand(N)

y = np.random.rand(N)

colors = np.random.rand(N)

size = (30 * np.random.rand(N))**2  # 0 to 15 point radii



plt.scatter(x, y, s=size, c=colors, alpha=0.5)

plt.show()
x=two_df[['priceUsd']]

y=two_df[['marketcap']]

two_df[['priceUsd','marketcap']].plot(kind='scatter', x='priceUsd', y='marketcap', alpha=0.5)
x=three_df[['priceUsd']]

y=three_df[['marketcap']]

three_df[['priceUsd','marketcap']].plot(kind='scatter', x='priceUsd', y='marketcap', alpha=0.5, legend=True) 
x=three_df[['priceUsd']]

y=three_df[['marketcap']]

three_df[['priceUsd','marketcap']].plot(kind='scatter', x='priceUsd', y='marketcap', alpha=0.5, legend=True,  s=size, figsize=(12,8))
hist=two_df.hist(column='priceUsd')

hist=three_df.hist(column='priceUsd')

hist=three_df.hist(figsize=(12,8))
hist = three_df.hist(column='priceUsd', bins=10, grid=False, figsize=(12,8), color='#4290be', zorder=2, rwidth=0.9)



hist = hist[0] # each unique value is accessed by its index (priceUsd) which is in clumn 0



for x in hist:



    # Switch off tickmarks

    x.tick_params(axis="both", which="both", bottom=False, top=True, labelbottom=True, left=False, right=False, labelleft=True)



    # Draw horizontal axis lines

    vals = x.get_yticks()

    for tick in vals:

        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)



    # Set title (set to "" for no title!)

    x.set_title("Price of BTC")



    # Set x-axis label

    x.set_xlabel("Price of BTC", labelpad=20, weight='bold', size=12)



    # Set y-axis label

    x.set_ylabel("Count?", labelpad=20, weight='bold', size=12)