# Importing modules
import pandas as pd
import numpy as np
import bq_helper
# Creating Big Query Helper object
hacker_news = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', dataset_name = 'hacker_news')
# List of tables from dataset
hacker_news.list_tables() 
# Printing schema from table "full"
hacker_news.table_schema("full")
# Print head of table "full"
hacker_news.head('full', selected_columns = 'by', num_rows = 10)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.head('global_air_quality')
# Creating query for exercise 1
query = """
    SELECT *
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit != 'ppm'
"""
# Estimating costs
open_aq.estimate_query_size(query)
# Extracting data to pandas
open_aq_unit = open_aq.query_to_pandas_safe(query, max_gb_scanned=1)
# Getting countries
countries = open_aq_unit.country.unique()
print(countries)
# Creating query for exercise 2
new_query = """
    SELECT *
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE value = 0;
"""
# Estimating costs
open_aq.estimate_query_size(query)
# Extracting data to pandas
open_aq_pol = open_aq.query_to_pandas_safe(new_query, max_gb_scanned=1)
## Getting pollutants with 0 value
pollutant_zero = open_aq_pol.loc[:, 'pollutant'].unique()
print(pollutant_zero)
# Importing modules
import pandas as pd
import numpy as np
import bq_helper
from matplotlib import pyplot as plt
# Creating Big Query Helper object
hacker_news = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', dataset_name = 'hacker_news')
# Creating query
query = """
    SELECT type, COUNT(id) as count
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
"""
# Estimating costs
hacker_news.estimate_query_size(query)
# Extracting data to pandas then printing
hacker_news_df = hacker_news.query_to_pandas_safe(query, max_gb_scanned=1)
print(hacker_news_df)
# Creating query
query = """
    SELECT COUNT(id) as count_deleted
    FROM `bigquery-public-data.hacker_news.comments`
    GROUP BY deleted
    HAVING deleted = true
"""
# Estimating costs
hacker_news.estimate_query_size(query)
# Extracting data to pandas then printing
hacker_news_df = hacker_news.query_to_pandas_safe(query, max_gb_scanned=1)
print(hacker_news_df)
# Creating query
query = """
    SELECT COUNTIF(deleted = true) as count_deleted
    FROM `bigquery-public-data.hacker_news.comments`
"""
# Estimating costs
hacker_news.estimate_query_size(query)
# Extracting data to pandas then printing
hacker_news_df = hacker_news.query_to_pandas_safe(query, max_gb_scanned=1)
print(hacker_news_df)
# Creating query for bananas
query = """
    SELECT *
    FROM `bigquery-public-data.hacker_news.full`
    WHERE REGEXP_CONTAINS(text, r"banana") OR REGEXP_CONTAINS(title, r"banana")
"""
# Estimating costs
print(hacker_news.estimate_query_size(query))
# Extracting data to pandas
hacker_news_df = hacker_news.query_to_pandas_safe(query, max_gb_scanned=7)
# Setting datetime index
hacker_news_df = hacker_news_df.set_index(pd.to_datetime(hacker_news_df.timestamp))
# Plotting the "banana" trend sampling per month
hacker_news_df.id.resample('m').count().plot()
# Query for posts per day
query = """
    SELECT count(id) as count, DATE(timestamp) as date
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY date
"""
# Estimating costs
print(hacker_news.estimate_query_size(query))
# Extracting post per day to pandas
hacker_news_ppd = hacker_news.query_to_pandas_safe(query, max_gb_scanned=7)
# Setting datetime index
hacker_news_ppd = hacker_news_ppd.set_index(pd.to_datetime(hacker_news_ppd.date))
# Plotting the post per month
hacker_news_ppd.resample('m').plot()
# Excluding timezone
hacker_news_df.index.tz = None
hacker_news_ppd.index.tz = None
# Calculating banana por post percentage
hacker_news_perc = hacker_news_df.id.resample('m').count() / hacker_news_ppd['count']
# Plotting the percentage
hacker_news_perc.resample('m').plot()
# Mean of bananas per post
hacker_news_perc.mean()
## Percentage of days with at least one "banana"
hacker_news_bpp = hacker_news_df.id.resample('D').count() / hacker_news_ppd['count']
hacker_news_ppd[hacker_news_bpp.isnull() | hacker_news_bpp != 0]['count'].count() / hacker_news_ppd['count'].count()
# Importing modules
import pandas as pd
import numpy as np
import bq_helper
from matplotlib import pyplot as plt
# Creating Big Query Helper object
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="nhtsa_traffic_fatalities")
# Query for accidents per hour
query = """SELECT COUNT(consecutive_number) as accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) as hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY accidents DESC
        """
# Estimating costs
print(accidents.estimate_query_size(query))
# Extracting post per day to pandas
accidents_df = accidents.query_to_pandas_safe(query, max_gb_scanned=1)
print(accidents_df)
# Query for hit and runs per stats
query = """SELECT COUNT(hit_and_run) as hit_and_runs, registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY hit_and_runs DESC
        """
# Estimating costs
print(accidents.estimate_query_size(query))
# Extracting post per day to pandas
hit_and_runs_df = accidents.query_to_pandas_safe(query, max_gb_scanned=1)
print(hit_and_runs_df) 