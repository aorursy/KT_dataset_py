# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# query to select all cities where
# unit is not equal to PPM
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'
            ORDER BY country
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
ppm_countries = open_aq.query_to_pandas_safe(query)

# display query results
ppm_countries
# query to select all pollutants that have had a reading where pollutant value
# where pollutant value was equal to zero with the location and values.
query = """SELECT DISTINCT pollutant, location, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant, location, value
        """
zero_pol_readings = open_aq.query_to_pandas_safe(query)
# display query results of pollutant by location and value.
zero_pol_readings