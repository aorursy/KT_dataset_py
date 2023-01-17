# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows 
open_aq.head("global_air_quality")
# query to select countries that use a diferrente unit from ppm to measure air quality
query = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
countries = open_aq.query_to_pandas_safe(query)
countries.head()
query="""SELECT distinct pollutant 
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value=0 """
pollutants = open_aq.query_to_pandas_safe(query)
pollutants.head()
