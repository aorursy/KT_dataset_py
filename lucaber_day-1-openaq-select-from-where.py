import bq_helper   # import package  with helper funcions

# create a helper object for this dataset
open_aq=bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")
open_aq.list_tables() #print all tables in the dataset

open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """
            SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.head()
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
us_cities.city.head().value_counts()
open_aq.head("global_air_quality",20)
## Which countries use a unit other than ppm to measure any type of pollution?
query = """
            SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

interesting_countries = open_aq.query_to_pandas_safe(query)
interesting_countries.head()
## Which pollutants have a value of exactly 0?
query = """
            SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

interesting_pollutants = open_aq.query_to_pandas_safe(query)
interesting_pollutants.head()
