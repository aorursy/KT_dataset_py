import bq_helper
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema("crime") #gives description of a specific table, in this case the table is called "crime"

chicago_crime.head("crime")
#chiago_crime.head("crime", selected_columns="name", num_rows=10) select specific column
query = """SELECT beat
            FROM `bigquery-public-data.chicago_crime.crime`
            WHERE location_description = "STREET" """
chicago_crime.estimate_query_size(query)
beat_on_street= chicago_crime.query_to_pandas_safe(query)
beat_on_street.beat.mean()
