# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# Your Code Goes Here
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
other_than_ppm = open_aq.query_to_pandas_safe(query1)
other_than_ppm.head()
other_than_ppm['country'].unique()
# Your Code Goes Here
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
zero_value = open_aq.query_to_pandas_safe(query2)
zero_value.head()
zero_value['pollutant'].unique()