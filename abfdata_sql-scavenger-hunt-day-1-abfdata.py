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
print(query)
# display multiple print results on one line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.head()
us_cities.tail()
us_cities.describe()
us_cities.shape
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# query countries that use a unit other than ppm to measure any type of pollution


unit_query = """SELECT country
                      FROM `bigquery-public-data.openaq.global_air_quality`
                      WHERE unit != 'ppm'
                  """
print(unit_query)
# pass pollution_query through open_aq.query_to_pandas_safe 
non_ppm = open_aq.query_to_pandas_safe(unit_query)
non_ppm.head()
non_ppm.tail()
non_ppm.describe()
non_ppm.shape
# get value_counts for country
non_ppm['country'].value_counts().head()
# which pollutants have a value of 0
pollution_val_zero = """SELECT pollutant, value
                         FROM `bigquery-public-data.openaq.global_air_quality`
                         WHERE value = 0
                     """
print(pollution_val_zero)
# pass pollution_val_zero through open_aq.query_to_pandas_safe 
val_zero = open_aq.query_to_pandas_safe(pollution_val_zero)
val_zero.head()
val_zero.tail()
val_zero.describe()
val_zero.shape
# get value counts for pollutants that have 0 value
val_zero['pollutant'].value_counts().head()