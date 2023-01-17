import pandas as pd 

import matplotlib.pyplot as plt

import sqlite3
# Connect to database

conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')
df = pd.read_sql_query("SELECT * From Fires", conn)

df.head()
# Create new DataFrame Populated with Relevant Data

df_year = df[['FIRE_YEAR']]
df_year['FIRE_YEAR'].value_counts(sort=False).plot(kind="line",marker='o', figsize=(8,5))

plt.title('Frequency of Wildfires Over Time')

plt.xlabel('Year')

plt.ylabel('Number of Instances')

plt.show()
df_size = df[['FIRE_SIZE']]

df_size = df_size.sort_values("FIRE_SIZE")



# Determine mean value

df_size["FIRE_SIZE"].mean()
# Histogram plot of all instances 

df_size.plot(kind="hist")

plt.title('Distribution of Fire Sizes')

plt.xlabel('Fire Size(Acres)')

plt.ylabel('Number of Instances')

plt.show()
# Bar plot with FIRE_SIZE_CLASS grouping

df_size_class = df[['FIRE_SIZE_CLASS']]

df_size_class.groupby('FIRE_SIZE_CLASS').size().plot(kind='bar', figsize=(8,5))

plt.title('Distribution of Fire Size Classes')

plt.xlabel('Fire Size Class')

plt.ylabel('Number of Instances')

plt.show()