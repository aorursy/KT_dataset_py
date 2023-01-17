# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas
data_path = '/kaggle/input/dwdm-week2-visualizations/US GDP.csv'
us_gdp = pd.read_csv(data_path)

us_gdp
len (us_gdp)


us_gdp.count()
#showws the first 3 of the data set 

us_gdp.head(3)
#shows the last 3 of the dataset

us_gdp.tail(3)
us_gdp['Year']
us_gdp['Year'].min()
#the row with index 1 

us_gdp.iloc[1]
us_gdp.iloc[1:5]
us_gdp.plot(kind = 'scatter',x= 'Year', y= 'US_GDP_BN',

           title= "US GDP per year",

           figsize=(12,8))



us_gdp.plot(kind = 'scatter',x= 'Year', y= 'US_GDP_BN',

           title= "US GDP per year",

           figsize=(12,8))

plt.title ("From %d to %d" % (

    us_gdp['Year'].min(),

    us_gdp['Year'].max()

))

plt.suptitle("US GDP per Year", size =12)

plt.ylabel("GDP")
#adding s  adjusts the scale ... dis nuh done yet

us_gdp.plot(kind = 'scatter',x= 'Year', y= 'US_GDP_BN',

            s=us_gdp[US_GDP_BN],

           title= "US GDP per year",

           figsize=(12,8))

plt.title ("From %d to %d" % (

    us_gdp['Year'].min(),

    us_gdp['Year'].max()

))

plt.suptitle("US GDP per Year", size =12)

plt.ylabel("GDP")
us_gdp.plot(kind='scatter',x='Year',y='US_GDP_BN',

            s= us_gdp['US_GDP_BN'].apply(lambda growth : 0 if growth < 0 else growth) * 0.09,

            title="US GDP per Year",

           figsize=(12,8))

plt.title("From %d to %d" % (

    us_gdp['Year'].min(),

    us_gdp['Year'].max()

),size=8)

plt.suptitle("US GDP per Year",size=12)

plt.ylabel("GDP")

"Size of Each point is GDP growth"