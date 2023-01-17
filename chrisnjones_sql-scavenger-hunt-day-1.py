# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
query = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
countries = open_aq.query_to_pandas_safe(query)
countries.country
query = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            where value = 0.0
        """
pollutants = open_aq.query_to_pandas_safe(query)
pollutants.pollutant