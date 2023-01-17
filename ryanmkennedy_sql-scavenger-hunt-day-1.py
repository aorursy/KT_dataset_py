# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
country_query = """
                    SELECT DISTINCT country 
                    FROM `bigquery-public-data.openaq.global_air_quality`   
                    WHERE unit <> 'ppm'
                    ORDER BY country
                """
countries = open_aq.query_to_pandas_safe(country_query)
countries
pollutant_query = """
                        SELECT DISTINCT pollutant 
                        FROM `bigquery-public-data.openaq.global_air_quality`   
                        WHERE value = 0
                        ORDER BY pollutant
                    """
pollutants = open_aq.query_to_pandas_safe(pollutant_query)
pollutants