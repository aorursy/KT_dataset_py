import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
query1 = """SELECT distinct country
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit!='ppm'
        """
q1_output = open_aq.query_to_pandas_safe(query1)
q1_output
query2 = """SELECT distinct pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value=0
        """
q2_output = open_aq.query_to_pandas_safe(query2)
q2_output