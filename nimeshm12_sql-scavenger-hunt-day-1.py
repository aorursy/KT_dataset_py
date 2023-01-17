import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()

open_aq.head("global_air_quality")
query1 = """
            SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
air_quality_countries = open_aq.query_to_pandas_safe(query1)
air_quality_countries.country.value_counts().head()

query2 = """
            SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
air_quality_pollutants = open_aq.query_to_pandas_safe(query2)
air_quality_pollutants.pollutant.value_counts().head()
