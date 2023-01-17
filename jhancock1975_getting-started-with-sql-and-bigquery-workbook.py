import bq_helper
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema('crime')
chicago_crime.head('crime')
query = """SELECT *
            FROM `bigquery-public-data.chicago_crime.crime`
            """

# check how big this query will be
help(chicago_crime.estimate_query_size)
print(chicago_crime.estimate_query_size(query))
chicago_crime.query_to_pandas_safe(query, max_gb_scanned=0.1)
query = """select description from `bigquery-public-data.chicago_crime.crime` where unique_key = 10152541"""
print(chicago_crime.estimate_query_size(query))