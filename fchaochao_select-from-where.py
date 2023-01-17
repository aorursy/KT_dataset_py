# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# Your Code Goes Here
query = """
        select country
        from `bigquery-public-data.openaq.global_air_quality`
        where unit != 'ppm'
        """
aq = open_aq.query_to_pandas_safe(query)
aq.head()
len(aq)
query = """
        select pollutant 
        from `bigquery-public-data.openaq.global_air_quality`
        where value = 0
        """
aq = open_aq.query_to_pandas_safe(query)
aq.head()