# Constants 

INPUT_PATH = '/kaggle/input/netflix-shows/netflix_titles.csv'



# Libraries 

import pandas as pd 

import matplotlib.pyplot as plt



# Set default properties for plotting 

plt.rcParams['figure.figsize'] = [11, 4]

plt.rcParams['figure.dpi'] = 100 
# Read data and display 5 random entries 

raw_df = pd.read_csv(INPUT_PATH)

raw_df.sample(5)
# Copy original data

df = raw_df.copy()



# Parse the raw `date_added` column to pandas datetime format

df['date_added'] = pd.to_datetime(df['date_added'])



# Check the type of the `date_added` column

df['date_added'].dtypes
# For each date, count the number of show added

# Note the regular usegae of `groupby` method

shows_added = df.groupby('date_added')[['show_id']].count()



# Rename column to describe the new content 

shows_added = shows_added.rename({'show_id': 

                                  'number_of_shows_added'}, 

                                 axis=1)



# View and plot TimeSeries 

shows_added.plot()

shows_added
# Add day name as a seperate column

shows_added['day_name'] = shows_added.index.day_name()



# Group by the extracted day name and sum up the 

total_per_weekday = shows_added.groupby('day_name')['number_of_shows_added'].sum()



# View results

# ... Seems that Netflix prefers to release the new shows on Fridays 

total_per_weekday
# Create mask of booleans from the required start date

# Note that the date is simple string and does not have to be a parsed datetime variable

mask = shows_added.index >= '2016-01-01'



# Apply the boolean mask to the dataframe 

shows_added = shows_added.loc[mask,:].copy()



# Plot the filtered data 

shows_added['number_of_shows_added'].plot()
# Use the resampling function to group data per week

weekly_data = (shows_added.

               resample('1W')   # For each week 

               .sum())          # Calculate the sum
# View and plot weekly data  

weekly_data.plot()

weekly_data