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

import matplotlib.pyplot as plt



#Opening the data file

df = pd.read_csv("../input/median-home-sale-price/Sale_Prices_City.csv")
#Viewing first 10 rows

df.head(10)
#Viewing last 10 rows

df.tail(10)
#Pulling up the list of all cities within the data

df['RegionName']
#Reshaping the data to move columns for each month into rows.  This allows me to be able to analyze the data.

#Update df after melt

df = pd.melt(df, id_vars=['RegionName'], value_vars=['2008-03', '2008-04', '2008-05', '2008-06', '2008-07', '2008-08', '2008-09', '2008-10', '2008-11', '2008-12',

                                                 '2009-01', '2009-02', '2009-03', '2009-04', '2009-05', '2009-06', '2009-07', '2009-08', '2009-09','2009-10', '2009-11', '2009-12',

                                                 '2010-01', '2010-02', '2010-03', '2010-04', '2010-05', '2010-06', '2010-07', '2010-08', '2010-09','2010-10', '2010-11', '2010-12',

                                                 '2011-01', '2011-02', '2011-03', '2011-04', '2011-05', '2011-06', '2011-07', '2011-08', '2011-09','2011-10', '2011-11', '2011-12',

                                                 '2012-01', '2012-02', '2012-03', '2012-04', '2012-05', '2012-06', '2012-07', '2012-08', '2012-09','2012-10', '2012-11', '2012-12',

                                                 '2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06', '2013-07', '2013-08', '2013-09','2013-10', '2013-11', '2013-12',

                                                 '2014-01', '2014-02', '2014-03', '2014-04', '2014-05', '2014-06', '2014-07', '2014-08', '2014-09','2014-10', '2014-11', '2014-12',

                                                 '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09','2015-10', '2015-11', '2015-12',

                                                 '2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09','2016-10', '2016-11', '2016-12',

                                                 '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09','2017-10', '2017-11', '2017-12',

                                                 '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09','2018-10', '2018-11', '2018-12',

                                                 '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09','2019-10', '2019-11', '2019-12'])
#Testing melt data prep

df
#Renaming columns and testing it. Update df after rename.

df = df.rename(columns = {'RegionName':'City', 'variable':'Date', 'value':'Median Home Sale Price'})

df
#Splitting Date column into individual Year and Month columns

df[['Year','Month']] = df.Date.str.split("-",expand=True)

df
#Sorting A to Z by City. Update df after sort.

#df = df.sort_values('City')

#df
#Filtering table down to values specifically for Seattle, ordered by date

seattle = df[(df.City == 'Seattle')]

#seattle.sort_values('Year')
#Creating a scatter plot to show the distribution of prices 

seattle_year = seattle['Year']

seattle_price = seattle['Median Home Sale Price']

plt.scatter(seattle_year, seattle_price)

plt.title('Distribution of Home Sale Prices in Seattle by Year')

plt.xlabel('Year')

plt.ylabel('Median Home Sale Price USD$')

#Creating a line chart to show the change prices for particular years

seattle_year = seattle['Date']

seattle_price = seattle['Median Home Sale Price']

plt.plot(seattle_year, seattle_price)

plt.title('Median Home Sale Prices in Seattle from 2008 - 2019')

plt.xlabel('Year', labelpad=20)

plt.ylabel('Median Home Sale Price USD$')