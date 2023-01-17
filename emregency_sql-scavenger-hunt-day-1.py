# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# build the query
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """
# run the query
non_ppm_countries = open_aq.query_to_pandas_safe(query)
#display
display(non_ppm_countries)


# build the query
query = """SELECT country, pollutant, SUM(value) AS total_pollution
            FROM `bigquery-public-data.openaq.global_air_quality`            
            GROUP BY country, pollutant
            HAVING total_pollution = 0
        """
# run the query
zero_pollutants = open_aq.query_to_pandas_safe(query)
#display
display(zero_pollutants)