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
# query to select all the items from the "country" column where the "unit" is not "ppm"
querynonppm = """SELECT country
                 FROM `bigquery-public-data.openaq.global_air_quality`
                 WHERE unit != 'ppm'
              """
# run the query in safe mode and store in a dataframe called non_ppm_countries
non_ppm_countries = open_aq.query_to_pandas_safe(querynonppm)
# list all non-ppm countries
non_ppm_countries.country.unique()
# query to select all the items from the "pollutant" column where the "value" is 0
querypollutant = """SELECT pollutant
                 FROM `bigquery-public-data.openaq.global_air_quality`
                 WHERE value = 0
              """
# run the query in safe mode and store in a dataframe called zero_pollutant
zero_pollutant = open_aq.query_to_pandas_safe(querypollutant)
# list all pollutants with a value of 0
zero_pollutant.pollutant.unique()