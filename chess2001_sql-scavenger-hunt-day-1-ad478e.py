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
#define non ppm query
query_non_ppm = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
            GROUP BY country
            ORDER BY country
        """
# check how big this query will be
open_aq.estimate_query_size(query_non_ppm)
#execute query. Only run this query if it's less than 10 MB
df_countries_non_ppm = open_aq.query_to_pandas_safe(query_non_ppm, max_gb_scanned=0.01)
df_countries_non_ppm.shape

#define query to retrieve pollutants with value equal to zero
query_zero_pollutants = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            GROUP BY pollutant
            ORDER BY pollutant
        """
#check how big this query will be
open_aq.estimate_query_size(query_zero_pollutants)
#get data
df_zero_pollutants = open_aq.query_to_pandas_safe(query_zero_pollutants, max_gb_scanned=0.01)
#output results
df_zero_pollutants