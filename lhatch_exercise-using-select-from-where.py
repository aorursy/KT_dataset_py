# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# select country from rows where pollution type measurement unit is not 'ppm'
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
countries_non_ppm_unit = open_aq.query_to_pandas_safe(query)
countries_non_ppm_unit.country.unique()
# alternate approach using query to get unique or distinct country list where unit is not 'ppm'
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            GROUP BY country
        """

countries_non_ppm_unit = open_aq.query_to_pandas_safe(query)
countries_non_ppm_unit
# assumption: questioner is interested in rows with zero values for any type of pollution
# select pollutant from rows where measurement unit is zero
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """

pollutants_measured_zero = open_aq.query_to_pandas_safe(query)
pollutants_measured_zero.pollutant.value_counts()