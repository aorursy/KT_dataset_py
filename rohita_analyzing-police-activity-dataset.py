import pandas as pd



#Read Data

ts = pd.read_csv("../input/traffic-stops-in-rhode-island/police.csv")

display(ts.head())



#Dataframe Info

display(ts.info())





#Look for missing values

display(ts.isnull().sum())

display(ts.shape)
# Drop state and county_name

display(ts['state'].value_counts())

ts.drop(['county_name','state'], axis='columns', inplace=True)



#Drop rows with NaN stop_date and stop_time

ts.dropna(subset=['stop_date', 'stop_time'], inplace=True)



display(ts.shape)
#Look at column data types

display(ts.dtypes)



#Change is_arrested to boolean

display(ts['is_arrested'].value_counts())

ts['is_arrested'] = ts['is_arrested'].astype('bool')



#Combine stop date and stop time to create pandas date time object

ts['stop_datetime'] = pd.to_datetime(ts['stop_date'].str.cat(ts['stop_time'], sep=" "))



display(ts.dtypes)



#Set datetime as index of dataframe

ts.set_index('stop_datetime', inplace=True)
#Examine Traffic Violations

display(ts['violation'].value_counts(normalize=True))



#Compare Violations by gender

female = ts[ts['driver_gender']=='F']

male = ts[ts['driver_gender']=='M']



display(female['violation'].value_counts(normalize=True))

display(male['violation'].value_counts(normalize=True))



#Single line

display(ts.groupby('driver_gender')['violation'].value_counts(normalize=True))
# Compare data by gender for speeding violation

female_speeding  = ts[(ts['driver_gender']=='F') & (ts['violation']=='Speeding')]

male_speeding  = ts[(ts['driver_gender']=='M') & (ts['violation']=='Speeding')]



display(female_speeding['stop_outcome'].value_counts(normalize=True))

display(male_speeding['stop_outcome'].value_counts(normalize=True))



#Single Line

ts[ts['violation']=='Speeding'].groupby('driver_gender')['stop_outcome'].value_counts(normalize=True)
# Compare data by gender on Vehicle Search

ts.groupby(['driver_gender'])['search_conducted'].mean()
# Compare search rate by gender and violation

ts.groupby(['violation','driver_gender'])['search_conducted'].mean()
# Use search type to look at rows with 'Protective Frisk'

display(ts['search_type'].value_counts())



# Since search type can have multiple values, containment need to be checked

ts['frisk'] = ts['search_type'].str.contains('Protective Frisk', na=False)



# Frisk condcuted by gender when vehicle was searched

display(ts[(ts['search_conducted'])].groupby('driver_gender')['frisk'].mean())

# Overall Arrest Rate

overall_arrest_rate = ts['is_arrested'].mean()



#Calculate hourly Arrest Rate. Resampling can also be used (used in next example)

hourly_arrest_rate = ts.groupby(ts.index.hour)['is_arrested'].mean()



# Plot Hourly Arrest Rate on a line chart

import matplotlib.pyplot as plt



# Create a line plot of 'hourly_arrest_rate'

plt.plot(hourly_arrest_rate)



# Add the xlabel, ylabel, and title

plt.xlabel("Hour")

plt.ylabel("Arrest Rate")

plt.title("Arrest Rate by Time of Day")



# Add reference line for overall Arrest Rate

plt.axhline(y=overall_arrest_rate, color='gray', linestyle='--')



# Display the plot

plt.show()
# Calculate annual trend by resampling data and plot line charts

annual_trend = ts.resample('A')[['drugs_related_stop','search_conducted']].mean()

annual_trend.plot(subplots=True)

plt.xlabel("Year")

plt.show()
# Create a crosstab and display a bar chart for each district

violations_by_zone =  pd.crosstab(ts['district'], ts['violation'])

violations_by_zone.plot(kind="bar",  stacked=True)

plt.show()



#Plot  only  for  K Zones

k_zones = violations_by_zone.loc['Zone  K1':'Zone K3']

k_zones.plot(kind="bar",  stacked=True)

plt.show()
# Print the unique values in 'stop_duration'

display(ts.stop_duration.unique())



# Create a dictionary that maps strings to integers

mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45} 



# Convert the 'stop_duration' strings to integers using the 'mapping'

ts['stop_minutes'] = ts.stop_duration.map(mapping)



# Print the unique values in 'stop_minutes'

display(ts['stop_minutes'].unique())



# Calculate the mean 'stop_minutes' for each value in 'violation_raw'

stop_length = ts.groupby('violation_raw')['stop_minutes'].mean()



# Sort 'stop_length' by its values and create a horizontal bar plot

stop_length.sort_values().plot(kind='barh')



# Display the plot

plt.show()
# Read weather dataset

weather = pd.read_csv("../input/weather-in-providence-rhode-island/weather.csv")

display(weather.head())



# Examine temperature columns for sanity: TMIN, TAVG, TMAX

display(weather[['TMIN','TAVG','TMAX']].describe())



# Visually examine with box plot

weather[['TMIN','TAVG','TMAX']].plot(kind="box")

plt.show()



# Ensure TMIN is always less than TMAX

# Create a 'TDIFF' column that represents temperature difference

weather['TDIFF'] = weather['TMAX'] - weather['TMIN']



# Describe the 'TDIFF' column

print(weather['TDIFF'].describe())



# Create a histogram with 20 bins to visualize 'TDIFF'

weather['TDIFF'].plot(kind='hist', bins=20)

plt.show()
# Copy 'WT01' through 'WT22' to a new DataFrame. Each column is a flag indicating bad weather

WT = weather.loc[:,'WT01':'WT22']



# Calculate the sum of each row in 'WT'

weather['bad_conditions'] = WT.sum(axis='columns')



# Replace missing values in 'bad_conditions' with '0'

weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')



# Create a histogram to visualize 'bad_conditions'

weather['bad_conditions'].plot(kind='hist')



# Display the plot

plt.show()
# Count the unique values in 'bad_conditions' and sort the index

print(weather['bad_conditions'].value_counts().sort_index())



# Create a dictionary that maps integers to strings

mapping = {0:'good', 1:'bad', 2:'bad', 3:'bad', 4:'bad', 5:'worse', 6:'worse', 7:'worse', 8:'worse', 9:'worse'}



# Convert the 'bad_conditions' integers to strings using the 'mapping'

weather['rating'] = weather['bad_conditions'].map(mapping)



# Count the unique values in 'rating'

print(weather['rating'].value_counts().sort_index())



# Create a list of weather ratings in logical order

cats = ['good', 'bad', 'worse']



# Change the data type of 'rating' to category

weather['rating'] = weather['rating'].astype(pd.api.types.CategoricalDtype(categories = cats))



# Examine the head of 'rating'

print(weather['rating'].head())
# Reset the index of Policing dataset

ts.reset_index(inplace=True)

display(ts.head())



# Create a DataFrame from the 'DATE' and 'rating' columns

weather_rating = weather[['DATE', 'rating']]

display(weather_rating.head())



#Merge the datasets

ts_weather = pd.merge(left=ts, right=weather_rating, left_on='stop_date', right_on='DATE', how='left')

display(ts_weather.head())
# Overall Arrest Rate

overall_arrest_rate = ts_weather['is_arrested'].mean()

display(overall_arrest_rate)



# Arrest Rate by weather rating

weather_arrest_rate = ts_weather.groupby('rating')['is_arrested'].mean()

display(weather_arrest_rate)



# Arrest Rate by Violation and weather rating

weather_violation_arrest_rate = ts_weather.pivot_table(index='violation', columns='rating', values='is_arrested')

display(weather_violation_arrest_rate)