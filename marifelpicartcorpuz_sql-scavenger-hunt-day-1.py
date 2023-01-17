import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
query1 = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
us_pollutant = open_aq.query_to_pandas_safe(query1)
print(us_pollutant)
query2 = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """


us_ctry = open_aq.query_to_pandas_safe(query2)
print(us_ctry)