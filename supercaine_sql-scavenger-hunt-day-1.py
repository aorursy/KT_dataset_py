# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")

# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
#What countries do use ppm as the method - using distinct SQL

PPM_QUERY = ("""
SELECT distinct(country)
FROM `bigquery-public-data.openaq.global_air_quality` 
WHERE unit = 'ppm'
""")

PPM_DF = open_aq.query_to_pandas_safe(PPM_QUERY)
PPM_DF
#Extract the countries which are not using PPM - value counts


NON_PPM_QUERY = ("""
SELECT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
""")

NON_PPM_DF = open_aq.query_to_pandas_safe(NON_PPM_QUERY)
NON_PPM_DF ['country'].unique()
open_aq.head("global_air_quality")
#Which pollutants have a value of exactly 0?
ZERO_POLLUTANT_QUERY = ("""

SELECT distinct(pollutant)
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0

""")

ZERO_P_DF = open_aq.query_to_pandas(ZERO_POLLUTANT_QUERY)
ZERO_P_DF
