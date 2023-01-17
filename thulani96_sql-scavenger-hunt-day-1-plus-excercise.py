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
# Import bq_helper
from bq_helper import BigQueryHelper
# Create helper object and connect to database
open_aq = BigQueryHelper('bigquery-public-data', 'openaq')
open_aq.list_tables()
# Look at the table details by each columns description
open_aq.table_schema('global_air_quality')
open_aq.head('global_air_quality')
data1 = """ 
            SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE `unit` != 'ppm'
        """

open_aq.estimate_query_size(data1) # Get Query Size
# Run the query and save results to a pandas dataframe
results1 = open_aq.query_to_pandas(data1) 

# Save the dataframe to CSV
results1.to_csv('countries_not_in_ppm_unit.csv')
data2 = """ 
            SELECT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE `value` = 0
        """

open_aq.estimate_query_size(data2)
result2 = open_aq.query_to_pandas(data2)
result2.to_csv('zero_value_pollutants.csv')