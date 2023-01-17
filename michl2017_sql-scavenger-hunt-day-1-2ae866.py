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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
#Question 1: Which countries use a unit other than ppm to measure any type of pollution?
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            GROUP BY country
        """
countries_not_using_ppm = open_aq.query_to_pandas_safe(query)
## check structure of new data frame countries_not_using_ppm
countries_not_using_ppm.shape
## The output tells us there are 64 counties which do not record their measurements in ppm (units).
## To display all 64 countries execute the command countries_not_using_ppm
## If we would like to look at only the first five countries in our dataframe (countries_not_using_ppm) 
## we would execute the command countries_not_using_ppm.head() 
countries_not_using_ppm
# Question 2: Which pollutants have a value of exactly 0?
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            GROUP BY pollutant
        """
pollutant_with_value_not_zero = open_aq.query_to_pandas_safe(query)
pollutant_with_value_not_zero.shape
#  The output shows there are 7 different pollutants, let’s display these.
pollutant_with_value_not_zero
