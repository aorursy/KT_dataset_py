# Load pandas

import pandas as pd



# Read in dataset

trends = pd.read_csv('../input/trends_kj_sisters.csv')



# Inspect data

trends.head()
# Make column names easier to work with

trends.columns = ['month', 'kim', 'khloe', 'kourtney', 'kendall', 'kylie']



# Inspect data

trends.head()
# Inspect data types

trends.info()
# Loop through columns

for column in trends.columns:

    # Only modify columns that have the "<" sign

    if "<" in trends[column].to_string(): 

        # Remove "<" and convert dtype to integer

        trends[column] = trends[column].str.replace("<","")

        trends[column] = pd.to_numeric(trends[column])



# Inspect data types and data

trends.info()

trends.head()
# Convert month to type datetime

trends['month'] = pd.to_datetime(trends.month)



# Inspect data types and data

trends.info()

trends.head()
# Set month as DataFrame index

trends = trends.set_index('month')



# Inspect the data

trends.head()
# Plot search interest vs. month

%matplotlib inline

trends.plot()
# Zoom in from January 2014

trends.loc['2014-01-01':'2019-03-31'].plot()
# Smooth the data with rolling means

trends.rolling(window=12).mean().plot()
# Average search interest for each family line

trends['kardashian'] = trends.kim + trends.khloe + trends.kourtney

trends['jenner'] = (trends.kendall + trends.kylie) / 2



# Plot average family line search interest vs. month

trends[['kardashian', 'jenner']].plot()