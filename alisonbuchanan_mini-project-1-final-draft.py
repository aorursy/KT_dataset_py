# import relevant libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# this code tells us what files we are working with and their names

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# We are working with two datasets: rainfall and library data. I will start by cleaning up the rainfall data. 

# To begin, I will save the csv rainfall data as a dataframe. 



rainfall_df = pd.read_csv('/kaggle/input/seattle-monthly-rainfall-data/Observed_Monthly_Rain_Gauge_Accumulations_-_Oct_2002_to_May_2017.csv')

rainfall_df.tail()
# Next, check the datatypes so we know what we are working with

rainfall_df ['Date'].dtypes
# I want to change the 'Date'column data to a string and only include the month and year

# to do this, I first need to change the data type to date time

rainfall_df['Date'] = pd.to_datetime(rainfall_df['Date'])



# Next, I need to create new columns to store the year and month

rainfall_df['year'] = rainfall_df['Date'].dt.year

rainfall_df['month'] = rainfall_df['Date'].dt.month



# check to see that worked

rainfall_df.head() 
# cast the new columns to a string datatype - this will make the data easier to work with 

rainfall_df['year'] = rainfall_df['year'].astype(str)

rainfall_df['month'] = rainfall_df['month'].astype(str)
# merge the year and month strings together and store them in a new "year_month" column

rainfall_df['year_month'] =  rainfall_df['year'].str.cat(rainfall_df['month'],sep="-")



# check to make sure this worked

rainfall_df.head()
# Rainfall is collected at multiple sites. For analysis, we just need the mean of all of those sites



# We can get monthly mean of all rain collection locations and add them to a newly created "average_rain" column 

rainfall_df['average_rain'] = rainfall_df.mean(axis=1,skipna =True, numeric_only = True)



rainfall_df.head()
# Now let's get rid of those individual collection site columns and unnecesary date columns



# We only need the year_month dates and rainfall means

rainfall_df = rainfall_df[['year_month','average_rain']]



rainfall_df.head()
# now we need to clean up the library data



# I will save the csv library data as a dataframe called lib_df. 

lib_df = pd.read_csv("/kaggle/input/seattle-public-library-checkout-data/Checkouts_by_Title.csv", usecols= ['CheckoutYear','CheckoutMonth','Checkouts'])



lib_df.tail()
# Next I will get rid of checkout dates from 2018 or later



# This will help us match the dates to the rainfall data

lib_df = lib_df[(lib_df['CheckoutYear'] < 2018)]



# Sort values to check that it worked

lib_df.sort_values(by = ['CheckoutYear'],inplace=True, ascending=False)

lib_df.head()
# Just like we did with the rainfall data, 

# We now need to turn the checkout year and month columns into strings 

# So we can more easily string them together as one date



lib_df['CheckoutYear'] = lib_df['CheckoutYear'].astype(str)

lib_df['CheckoutMonth'] = lib_df['CheckoutMonth'].astype(str)



# check the datatypes to make sure this worked

lib_df.dtypes
# merge the checkout year and month together and store them in a new checkout_datetime column 

lib_df['year_month'] = lib_df['CheckoutYear'].str.cat(lib_df['CheckoutMonth'],sep="-")



lib_df.head()
# clean up the dataframe so it only contains the columns we will need for analysis

lib_df = lib_df[['Checkouts','year_month']]



lib_df.head(15)
# Now it is time to figure out how many items total were checked out for each date

# There is a pandas group function to do this!

# Update the datafile to group the duplicate checkout_datetime data 

# Add together all of the associated "checkouts" data and include the sum in the checkouts column



lib_df = lib_df.groupby('year_month', as_index=False).agg({"Checkouts": "sum"})

lib_df.head()
# First we need to merge to two dataframes so they can be plotted on the same figure

# Because the dataframes share the same 'year-month'colmun data

# I can simply merge the dataframes and specify that the data should

# remain affiliated with their 'year month' info



df_lib_rain = pd.merge(lib_df, rainfall_df, on='year_month')

df_lib_rain.head()
# Next, we want to normalize the data so we can look at rainfall and checkouts

# on a similar scale and better compare them



df_lib_rain["Checkouts"] = df_lib_rain["Checkouts"] / df_lib_rain["Checkouts"].max()

df_lib_rain["average_rain"] = df_lib_rain["average_rain"] / df_lib_rain["average_rain"].max()
# finally we plot the data on a line graph over time to see how the vlaues might interact

df_lib_rain.plot(x="year_month")
# Lets see what a bar graph would look like - just for fun



df_lib_rain.plot.bar(x='year_month',figsize = (30,10))