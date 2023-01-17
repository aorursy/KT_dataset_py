import bq_helper
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema('crime')
chicago_crime.head("crime")
chicago_crime.head("crime",selected_columns="description", num_rows=2)
query = """SELECT description
            FROM `bigquery-public-data.chicago_crime.crime`
            WHERE description = "POCKET-PICKING" """
chicago_crime.estimate_query_size(query)
pocket_picking = chicago_crime.query_to_pandas_safe(query)
