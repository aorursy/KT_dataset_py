import bq_helper

# create a helper object for our bigquery dataset

chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema("crime")
chicago_crime.head("crime")
chicago_crime.head("crime",selected_columns="description",num_rows=10)
help(chicago_crime)
# BigQuery Standard SQL requires backticks around the table name ``

query = """SELECT count(unique_key)

            FROM `bigquery-public-data.chicago_crime.crime`

            """



chicago_crime.estimate_query_size(query)
no=chicago_crime.query_to_pandas_safe(query, max_gb_scanned=0.1)
no
query_1 = """SELECT date

                FROM `bigquery-public-data.chicago_crime.crime`

                WHERE district = 9"""

chicago_crime.estimate_query_size(query_1)
date_crime = chicago_crime.query_to_pandas_safe(query_1,max_gb_scanned=0.2)
date_crime