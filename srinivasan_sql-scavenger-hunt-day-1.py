# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
#countries use a unit other than ppm to measure any type of pollution
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
countries_not_using_ppm_unit = open_aq.query_to_pandas_safe(query)
countries_not_using_ppm_unit.country

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

#Which pollutants have a value of exactly 0
query_zero_pollutant = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """
zero_pollutant = open_aq.query_to_pandas_safe(query_zero_pollutant)
zero_pollutant.pollutant

