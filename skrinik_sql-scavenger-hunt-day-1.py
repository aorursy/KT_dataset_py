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
# import our bq_helper package
import bq_helper 

#import data
bg_data =  bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")

#look at structure
bg_data.list_tables()
bg_data.table_schema("global_air_quality")
#Query 1: Which countries use a unit other than ppm to measure any type of pollution? 
#(Hint: to get rows where the value *isn't* something, use "!=")
query1 = """
select distinct country
from `bigquery-public-data.openaq.global_air_quality`
where unit != 'ppm'
"""
bg_data.query_to_pandas_safe(query1).head()
#Query 2: Which pollutants have a value of exactly 0?
query2 = """
select location, country, pollutant
from `bigquery-public-data.openaq.global_air_quality`
where value = 0
"""
bg_data.query_to_pandas_safe(query2).head()
#save output & show head
nonPPM_cities = bg_data.query_to_pandas_safe(query1, max_gb_scanned=0.1)
pollutant_zero = bg_data.query_to_pandas_safe(query2, max_gb_scanned=0.1)


#Write to csv, query 1 *& 2 respectively:
nonPPM_cities.to_csv("nonPPM_cities.csv")
pollutant_zero.to_csv("pollutant_zero.csv")