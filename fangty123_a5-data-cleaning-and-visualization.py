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
# To analyze the dataset and answer questions I asked above with data visualization,

# the first step is to load the data stored in CSV and clean it using pandas DataFrame.



listings = pd.read_csv('/kaggle/input/airbnb-seattle-listing-nov-2019/listings.csv')



# print the number of listings and types of metadata

print('Number of listings: ' + str(listings.shape[0]))

print('Types of metadata: ' + str(listings.shape[1]))



# sample top 5 lines of data to see what's available

# also force pandas to display all columns

pd.set_option('display.max_columns', 110)

listings.head()
# select the relevant data from overall dataset

cleaned_data = listings.loc[:, ['neighbourhood', 'price', 'has_availability']]



# Next, sample the last 25 rows of data to make sure it contains correct information

cleaned_data.tail(25)



# confirm the shape of cleaned data

print(cleaned_data.shape)
# from the sample, I saw 2 issues:

# 1. Some data rows has neighbourhood == NaN, which needs to be filtered out

cleaned_data = cleaned_data.dropna()

cleaned_data.tail(10)



# confirm the shape of cleaned data - 1 row was removed

print(cleaned_data.shape)
# 2. It looks like all data rows has has_availability == t, and if that's the case my 

# question 3 (Which area/neighbourhood is most in demand (lowes level of availablity) 

# for Airbnb listing?) cannot be answered. Need to examine filtered data to confirm



availability = cleaned_data['has_availability'] != 't'

availability_data = cleaned_data[availability]

availability_data.tail(10)
# And as expected, there is no data rows in the data frame that has has_availability = 'f'

# As a result, I decided to use column availability_60 and availability_365 for this purpose.

# My plan is to plot based on availability_60, but use availability_365 for data cleaning.



cleaned_data = listings.loc[:, ['neighbourhood', 'price', 'availability_60', 'availability_365']]

cleaned_data = cleaned_data.dropna()



# clearly, listing that has availability_365 == 0 does not make sense (full year booked, really?)

# filter them out from this analysis

valid_listing = cleaned_data['availability_365'] != 0

valid_listing_data = cleaned_data[valid_listing]



# confirm the shape of cleaned data - 2054 row was removed

print(valid_listing_data.shape)
# now, with data cleaned, I'm ready to start plotting



import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))



# 1. Which area/neighbourhood has the most number of Airbnb listing?



# 1a. count the number of listing by each neighbourhood

listing_num = valid_listing_data.groupby('neighbourhood')["price"].count().reset_index(name="count")



# 1b. try plot a pie chart

listing_num.plot.pie(y='count', figsize=(10, 10), legend=False)



# 1c. that's not very easy to read, because there are a lot of neighbourhoods

# select the top 50 after sorting for bar chart

listing_num = listing_num.sort_values(by='count', ascending=False).head(50)



# 1d. plot the bar chart and make sure it's big enough to read

listing_num.plot.bar(x='neighbourhood', y='count', figsize=(20, 10))
# 2. Which area/neighbourhood has the highest average price for Airbnb listing?



# 2a. try plot the histogram of price, which requires converting the price into float first

# The chart is skewed due to some ultra-high price listing

price_data = valid_listing_data['price'].replace('[\$,]', '', regex=True).astype(float)

price_data.plot.hist(bins=500, figsize=(20, 10))



# 2b. average the price by each neighbourhood, note I used median to offset those ultra-high price listings

# again select the top 20 after sorting for bar chart

valid_listing_data['price'] = price_data

ne_price_data = valid_listing_data.groupby('neighbourhood')["price"].median().reset_index(name="mean_price")

ne_price_data = ne_price_data.sort_values(by='mean_price', ascending=False).head(50)



# 2c. plot the bar chart and make sure it's big enough to read

ne_price_data.plot.bar(x='neighbourhood', y='mean_price', figsize=(20, 10))
# 3. Which area/neighbourhood is most in demand (lowes level of availablity) for Airbnb listing?



ava_data = valid_listing_data.groupby('neighbourhood')["availability_60"].mean().reset_index(name="avg_availability_60")

ava_data = ava_data.sort_values(by='avg_availability_60').head(50)

ava_data.plot.bar(x='neighbourhood', y='avg_availability_60', figsize=(20, 10))
