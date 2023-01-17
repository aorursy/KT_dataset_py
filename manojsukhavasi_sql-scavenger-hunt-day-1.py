# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.table_schema("global_air_quality")
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality") 
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
open_aq.estimate_query_size(query)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.head(15)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
def get_pandas_df_from_query(query, bq_obj=open_aq):
    print("Estimated query size is {0:.2f} MB".format(bq_obj.estimate_query_size(query) * 1000))
    return open_aq.query_to_pandas_safe(query)
q1 = """SELECT DISTINCT unit
        FROM `bigquery-public-data.openaq.global_air_quality`
     """
q1df = get_pandas_df_from_query(q1)
q1df.head()
# Which countries use a unit other than ppm to measure any type of pollution? 
# Your code goes here :)
no_ppm_query = """SELECT DISTINCT country
                  FROM `bigquery-public-data.openaq.global_air_quality`
                  WHERE unit != 'ppm'
               """
no_ppm_countries = get_pandas_df_from_query(no_ppm_query)
no_ppm_countries.count()
# Which pollutants have a value of exactly 0?
# Your code goes here :)
zero_poll_query = """SELECT DISTINCT pollutant
                  FROM `bigquery-public-data.openaq.global_air_quality`
                  WHERE value = 0.00
               """
zero_pollutants = get_pandas_df_from_query(zero_poll_query)
zero_pollutants.head(10)