# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

#open_aq.head("global_air_quality")

## Query for countries having pollution unit not ppm
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'
            ORDER BY country
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
uniq_countries = open_aq.query_to_pandas_safe(query)
uniq_countries

## Query for pollutant with 0 value
query_pollutant = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
        """
pollutant_zero = open_aq.query_to_pandas_safe(query_pollutant)
#pollutant_zero
