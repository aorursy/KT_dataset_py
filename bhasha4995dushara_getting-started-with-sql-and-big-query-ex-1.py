import bq_helper
import pandas as pd
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema("crime")
chicago_crime.head("crime")
chicago_crime.head("crime",selected_columns=["unique_key","case_number","date"],num_rows=10,start_index=2)
q = """select case_number from `bigquery-public-data.chicago_crime.crime` where district = 4 """
chicago_crime.estimate_query_size(q)
chicago_crime.query_to_pandas(q)
