# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")

my_query_1="""SELECT distinct country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            where unit <>"ppm" order by country
        """


countries = open_aq.query_to_pandas_safe(my_query_1)
countries
my_query_2= """ select distinct pollutant, value 
                FROM `bigquery-public-data.openaq.global_air_quality`
                where value=0
                order by pollutant 
                """
pollutant = open_aq.query_to_pandas_safe(my_query_2)
pollutant