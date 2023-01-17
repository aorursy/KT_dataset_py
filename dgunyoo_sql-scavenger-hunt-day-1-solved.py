# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

query1 = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            where unit != 'ppm'
        """

q1 = open_aq.query_to_pandas_safe(query1)

query2 = """SELECT distinct country, pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            where value = 0
            order by country
        """

q2 = open_aq.query_to_pandas_safe(query2)

print('Countries using units other than ppm\n',q1,'\n','Countries that have 0 pollutants',q2)