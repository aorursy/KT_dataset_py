# Manipulate Data
import pandas as pd
import numpy as np
# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
# Query
import bq_helper

# Connect to the database by using bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                      dataset_name = 'hacker_news')
# View Schema, check to see that you are scanning the right tables
hacker_news.list_tables()
# Calls information on all the columnn in the "full" table in the hacker_news database
# This takes the role of looking at the column metadata
hacker_news.table_schema('full')
# View a few results to get a general understanding of the data presented
hacker_news.head('full')
# Other functionality that is possible. 
hacker_news.head('full', selected_columns = 'by', num_rows = 10)
# Recall previously, we wrote
#hacker_news = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
#                                      dataset_name = 'hacker_news')

query = """
SELECT score
FROM `bigquery-public-data.hacker_news.full`
WHERE type = 'job'
"""

# This prints out the size of the query in megabytes. 
hacker_news.estimate_query_size(query)
# Another alternative that uses bigquery and pandas

# from google.cloud import bigquery
# import pandas as pd

# client = bigquery.Client()

# # Using WHERE reduces the amount of data scanned / quota used
# query = """
# SELECT title, time_ts
# FROM `bigquery-public-data.hacker_news.stories`
# WHERE REGEXP_CONTAINS(title, r"(k|K)aggle")
# ORDER BY time
# """

# query_job = client.query(query)

# iterator = query_job.result(timeout=30)
# rows = list(iterator)

# # Transform the rows into a nice pandas dataframe
# headlines = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# # Look at the first 10 headlines
# headlines.head(10)
# How to save the data from the query as a .csv

# df.to_csv('name.csv', encoding = 'utf-16')
#Redundant import
import bq_helper

# open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
#                                    dataset_name="openaq")
open_aq = bq_helper.BigQueryHelper("bigquery-public-data", "openaq")
open_aq.list_tables()
# From the listed tables, we see we can explore global_air_quality
open_aq.head('global_air_quality')
# Lets say we are only looking to explore the cities in the US
# Remember to use backticks
query = """
SELECT city
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE country = 'US'
"""
us_cities = open_aq.query_to_pandas_safe(query, max_gb_scanned = .1)
us_cities.city.value_counts().head()
# Lets explore this
# We need to find out which column has this information. 
# Find out the different units used
# Then get the cities that use the different units
open_aq.head('global_air_quality')
# It seems like we want to query the city and the units
query = """
SELECT country, unit
FROM `bigquery-public-data.openaq.global_air_quality`
"""
open_aq.estimate_query_size(query)
# 0.00019 gigabytes seems relatively small. I am fine querying that dataset
country_units = open_aq.query_to_pandas(query)
country_units.describe()
country_units.unit.value_counts()
# We can see that 13803 data entries use the unit other than ppm. Lets find out which ones
# I will use head() because there are too many entries
country_units.country[country_units.unit != 'ppm'].value_counts().head()
# It seems like every country in our dataset uses the unit that isnt ppm. That's anticlimatic. 
(len(country_units.country[country_units.unit != 'ppm'].value_counts()) / 
len(country_units.country.value_counts()))
# For the second question, we will explore which pollutants have a value of exactly 0
query = """
SELECT pollutant, value
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""
pollutants = open_aq.query_to_pandas(query)
pollutants.pollutant.value_counts()
# This would be another way to do it all in SQL
query = """
SELECT pollutant, COUNT(value)
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
GROUP BY pollutant
"""
open_aq.query_to_pandas(query)
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query = """
SELECT parent, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY parent
HAVING COUNT(id) > 10
"""
# This result gets the parent id, which then needs to be joined on the original dataset
# to find the corresponding text to each parent id. 
hacker_news.query_to_pandas_safe(query, max_gb_scanned=.1)
import bq_helper
traffic = bq_helper.BigQueryHelper('bigquery-public-data', 'nhtsa_traffic_fatalities')
# Limit the results to only show 5
traffic.list_tables()[:5]
traffic.head('accident_2015')
query = """
SELECT EXTRACT(HOUR FROM timestamp_of_crash), COUNT(consecutive_number) as num_of_acc
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
ORDER BY COUNT(consecutive_number) DESC
"""

traffic.estimate_query_size(query)
df = traffic.query_to_pandas(query)
df
traffic.head('vehicle_2015', 
             selected_columns = ['registration_state_name', 'hit_and_run'],
            num_rows = 10
            )
query = """
SELECT registration_state_name, COUNT(consecutive_number)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
WHERE hit_and_run != 'No'
GROUP BY registration_state_name
ORDER BY COUNT(consecutive_number) DESC
"""

traffic.estimate_query_size(query)
df = traffic.query_to_pandas(query)
df
