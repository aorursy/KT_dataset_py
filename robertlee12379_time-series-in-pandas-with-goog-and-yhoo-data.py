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
# Compare annual stock price trends

# We are going to compare the performance for three years of Yahoo stock prices.

yahoo = pd.read_csv('../input/yahoostock/yahoo.csv')



# Explore the dataset

yahoo.head()

yahoo.info()



# Convert the date column to datetime64

# dataframe['column_name'] = pd.to_datetime(dataframe['column_name']) is equivalent to parse_dats = ['column_name']

yahoo['date'] = pd.to_datetime(yahoo['date'])



# Set date column as index

yahoo.set_index('date', inplace=True)



# Plot data

yahoo.plot(title = "Yahoo's Stock Price from 2013 to 2015")

plt.show()



# Create dataframe prices here

prices = pd.DataFrame()



# Select data for each year and concatenate with prices here 

# Iterate over a list containing the three years, 2013, 2014, and 2015, as string, and in each loop:

price_per_year = yahoo.loc['2013', ['price']].reset_index(drop=True)

for year in ['2013', '2014', '2015']:

    

    # Use the iteration variable to select the data for this year and the column price

    # Use .reset_index() with drop=True to remove the DatetimeIndex

    price_per_year = yahoo.loc[year, ['price']].reset_index(drop=True)

    

    # Rename the column price column to the appropriate year

    price_per_year.rename(columns={'price': year}, inplace=True)

    

    # Use pd.concat() to combine the yearly data with the data in prices along axis=1 (column)

    prices = pd.concat([prices, price_per_year], axis=1)



# Plot prices

prices.plot(title = "Compare Yahoo's Stock Price 2013, 2014, and 2015")

plt.show()
# Percentage Change

# Let's compare a stock price series for Google shifted 90 business days into both past and future.



# Import data here

# Use pd.read_csv() to import 'google.csv', parsing the 'Date' as dates, setting the result as index and assigning to google.

# parse_dats = ['column_name'] is equivalent to dataframe['column_name'] = pd.to_datetime(dataframe['column_name'])

google = pd.read_csv('../input/googlestock/google.csv', parse_dates=['Date'], index_col='Date')



# Set data frequency to business daily

google = google.asfreq('B')



# Plot data

google.plot(title = "Google's Stock Price from 2015 to 2016")

plt.show()



# Add new columns lagged and shifted to google that contain the Close shifted by 90 business days into past and future, respectively

google['lagged'] = google.Close.shift(periods=-90)

google['shifted'] = google.Close.shift(periods=90)



# Rename the google Close column 

google.rename(columns = {'Close':'price'}, inplace = True)



# Plot the google price series

google.plot()

plt.show()
# Method 1

# Created shifted_30 here

google['shifted_30'] = google.price.shift(30)



# Subtract shifted_30 from price

google['change_30'] = google.price.sub(google.shifted_30)



# Method 2 

# Get the 30-day price difference

google['diff_30'] = google.price.diff(periods = 30)



# Inspect the last five rows of price

print(google.tail(40))



# Show the value_counts of the difference between change_30 and diff_30

print('\n')

print('When you substract google.diff_30 from google.change_30, the value is equal to 0')

print(google.change_30.sub(google.diff_30).value_counts())
# Plotting multi-period returns



# Create daily_return

google['daily_return'] = google.price.pct_change(1) * 100



# Create monthly_return

google['monthly_return'] = google.price.pct_change(30) * 100



# Create annual_return

google['annual_return'] = google.price.pct_change(360) * 100



# Drop columns 

drop_columns = ['lagged', 'shifted', 'shifted_30', 'change_30', 'diff_30']

google.drop(drop_columns, axis=1, inplace=True)



# Plot the result

google.plot(subplots=True)

plt.show()