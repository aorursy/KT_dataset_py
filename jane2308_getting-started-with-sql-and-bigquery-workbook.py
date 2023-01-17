import bq_helper
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema("crime")
chicago_crime.head("crime")
query = """SELECT location_description
            FROM `bigquery-public-data.chicago_crime.crime`
            """

chicago_crime.estimate_query_size()
chicago_crime.query_to_pandas_safe(query, max_gb_scanned=0.1)
year = chicago_crime.query_to_pandas_safe(query)
