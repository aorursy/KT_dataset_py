import bq_helper as bq

openaq= bq.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")

#To list all tables

openaq.list_tables()
#Schema details
openaq.table_schema("global_air_quality")
openaq.head("global_air_quality")

query = """SELECT city 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE country='US' """

#Estimating the size of query
openaq.estimate_query_size(query)
openaq_cities = openaq.query_to_pandas_safe(query)

openaq_cities.city.value_counts()

query1 = """SELECT distinct country 
           FROM `bigquery-public-data.openaq.global_air_quality` 
           WHERE unit!='ppm' """

openaq.estimate_query_size(query1)

openaq_country = openaq.query_to_pandas_safe(query1)
openaq_country.head()

query2 = """SELECT distinct pollutant
           FROM `bigquery-public-data.openaq.global_air_quality` 
           WHERE value = 0 """

openaq.estimate_query_size(query2)

openaq_zerovalue_pollutant=openaq.query_to_pandas_safe(query2)
openaq_zerovalue_pollutant
query3 = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality` """
openaq.estimate_query_size(query3)

df = openaq.query_to_pandas_safe(query3)