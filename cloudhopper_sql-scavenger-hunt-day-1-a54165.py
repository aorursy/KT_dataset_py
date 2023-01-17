# Your code goes here :)

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")


# query to select all the items from the "country" column where the
# "unit" column is not "ppm"
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
no_ppm = open_aq.query_to_pandas_safe(query)

# What five units have the most measurements taken there?
no_ppm.country.value_counts().head()


# query to select all the items from the "pollutant" column where the
# "value" column is 0 with the ' 'quotation removed
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
z_pollutant = open_aq.query_to_pandas_safe(query)

# What five pollutants have the most 0 taken there?
z_pollutant.pollutant.value_counts().head()