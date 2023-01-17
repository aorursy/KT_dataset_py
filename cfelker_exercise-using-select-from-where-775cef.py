# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'µg/m³'
        """
country = open_aq.query_to_pandas_safe(query)
country.country.value_counts().head()
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = '0.0'
        """
series = open_aq.query_to_pandas_safe(query)
series.pollutant.value_counts().head()