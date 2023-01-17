# Your code goes here :)

# import package with helper functions 
import bq_helper
import pandas as pd

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
            GROUP BY country
        """

ppm_pollutants = open_aq.query_to_pandas_safe(query)

# Print all countries
with pd.option_context('display.max_rows', None):
    display(ppm_pollutants)
# Which pollutants have a value of exactly 0?
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            GROUP BY pollutant
        """

zero_pollutants = open_aq.query_to_pandas_safe(query)

# Print all pollutants 
zero_pollutants.head(zero_pollutants.shape[0])
