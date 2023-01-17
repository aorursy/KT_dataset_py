import bq_helper

air_quality = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="openaq")

air_quality.list_tables()
air_quality.table_schema("global_air_quality")
air_quality.head("global_air_quality", selected_columns="country", num_rows=12)
air_quality.head("global_air_quality")
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' """

air_quality.estimate_query_size(query)
countries = air_quality.query_to_pandas_safe(query)
countries["country"].unique()
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """
pollutants = air_quality.query_to_pandas_safe(query2)
pollutants["pollutant"].unique()