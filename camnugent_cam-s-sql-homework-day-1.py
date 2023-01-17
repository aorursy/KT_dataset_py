import bq_helper

data = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                dataset_name='openaq')



data.list_tables()
data.head("global_air_quality")
query1 = """ SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'"""
data.estimate_query_size(query1)
q1_data = data.query_to_pandas_safe(query1)
q1_data
query2 = """SELECT location, city,  pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0"""
data.estimate_query_size(query2)
data2 = data.query_to_pandas_safe(query2)
data2