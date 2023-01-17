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

# create a helper object for our bigquery dataset
openaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                  dataset_name = "openaq")

# Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value isn't something, use "!=")
query3 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" 
         """
# only run this query3 if it's less than 1MB
countries_noppm = openaq.query_to_pandas_safe(query3, max_gb_scanned=0.001)
# Which are the countries which don't use ppm to measure any type of pollution
list_of_unique_country = countries_noppm.country.unique()
print("There are {0} countries that use another unit than ppm to measure any type of pollution: {1}".format(len(list_of_unique_country), list_of_unique_country))

# Which pollutants have a value of exactly 0?
query4 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 
         """
# only run this query4 if it's less than 1MB
pollutant_0 = openaq.query_to_pandas_safe(query4, max_gb_scanned=0.001)
# Which are the pollutants that have
list_of_unique_pollutant_0 = pollutant_0.pollutant.unique()
print("There are {0} pollutants that have a zero value: {1}".format(len(list_of_unique_pollutant_0), list_of_unique_pollutant_0))