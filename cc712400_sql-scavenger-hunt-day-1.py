# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# Identify what types of units are being used
query = """SELECT DISTINCT(unit)
            FROM `bigquery-public-data.openaq.global_air_quality`
        """

units = open_aq.query_to_pandas_safe(query)
print(units)
# Countries with at least one city with units other than ppm or µg/m³
query = """SELECT country AS countries_with_units_other_than_ppm
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            GROUP BY country
        """

countries_with_units_other_than_ppm = open_aq.query_to_pandas_safe(query)
print(countries_with_units_other_than_ppm[:5])
# Pollutants which have at least one entry of exactly 0
query = """SELECT pollutant AS pollutants_with_one_value_of_zero
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            GROUP BY pollutant
        """

pollutants_with_one_value_of_zero = open_aq.query_to_pandas_safe(query)
print(pollutants_with_one_value_of_zero)