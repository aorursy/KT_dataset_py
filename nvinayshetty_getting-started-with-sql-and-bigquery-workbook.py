import bq_helper
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data",dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema("crime")

chicago_crime.head("crime")
chicago_crime.head("crime",selected_columns="ward", num_rows=10)
query = """SELECT year
            FROM `bigquery-public-data.chicago_crime.crime`
            WHERE district = 8 """
chicago_crime.estimate_query_size(query)
years=chicago_crime.query_to_pandas_safe(query, max_gb_scanned=1)
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()