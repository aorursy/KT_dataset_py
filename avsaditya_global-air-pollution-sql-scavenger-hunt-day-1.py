import bq_helper
open_aq= bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")
open_aq.list_tables()
open_aq.head('global_air_quality')
query = """SELECT city  
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
us_cities=open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
not_ppm_query="""SELECT DISTINCT country
                 FROM `bigquery-public-data.openaq.global_air_quality`
                 WHERE unit != 'ppm'
              """
not_ppm_cities=open_aq.query_to_pandas_safe(not_ppm_query)
not_ppm_cities
zero_pollutant_query = """SELECT DISTINCT pollutant 
                          FROM `bigquery-public-data.openaq.global_air_quality`
                          WHERE value = 0.00
                       """
zero_pollutants=open_aq.query_to_pandas_safe(zero_pollutant_query)
zero_pollutants
