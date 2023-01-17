import bq_helper
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")

# chicago_crime.list_tables()
# chicago_crime.table_schema('crime')
# chicago_crime.head('crime')
# chicago_crime.head('crime',selected_columns='case_number',num_rows=10)

# query = """
#             SELECT fbi_code FROM `bigquery-public-data.chicago_crime.crime`
#             WHERE year = 2016
#         """
# chicago_crime.estimate_query_size(query)

# chicago_crime.query_to_pandas_safe(query,max_gb_scanned=.1)

# some_variable = chicago_crime.query_to_pandas_safe(query)

# some_variable.column_name.mean()