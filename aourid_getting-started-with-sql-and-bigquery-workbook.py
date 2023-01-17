import bq_helper
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema('crime')
chicago_crime.head('crime')
query = """SELECT case_number
            FROM `bigquery-public-data.chicago_crime.crime`
            WHERE primary_type = 'HOMICIDE' """
chicago_crime.estimate_query_size(query)

# Return a dataframe
case_numer_df = chicago_crime.query_to_pandas_safe(query)
case_numer_df.info()
case_numer_df.head(10)