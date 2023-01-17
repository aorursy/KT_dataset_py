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
# To answer first question: Find countries that don't include ppm in the column
# query to select all the items from the "country" column where the
# "pollutant" column does not include ppm as a unit.
pollution_query = """
            SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
            """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollution_units = open_aq.query_to_pandas_safe(pollution_query)
# Number of unique countries in the dataframe
pollution_units.country.nunique()
# prints array of unique countries in the dataframe
print(pollution_units.country.unique())
#Now to answer second question: find which values in pollutant column are equal to 0

poll_0_q = """
            SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            """
zero_pollution = open_aq.query_to_pandas_safe(poll_0_q)
print(zero_pollution.pollutant.unique())