# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# Your Code Goes Here
query="""SELECT country
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE unit != 'ppm'
       """
locs=open_aq.query_to_pandas_safe(query)
locs.country.unique()
# Your Code Goes Here
query2="""SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
pollutes=open_aq.query_to_pandas_safe(query2)
pollutes.pollutant.unique()