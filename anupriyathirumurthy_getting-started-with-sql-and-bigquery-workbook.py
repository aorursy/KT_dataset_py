import bq_helper
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
# print a list of all the tables in the chicago_crime dataset
chicago_crime.list_tables()
# print information on all the columns in the "crime" table
# in the chicago_crime dataset
chicago_crime.table_schema("crime")
# preview the first couple lines of the "crime" table
chicago_crime.head("crime")
# preview the first ten entries in the primary_type column of the crime table
chicago_crime.head("crime", selected_columns="primary_type", num_rows=10)
# this query looks in the crime table in the chicago_crime dataset, then gets the score column from every row where                  the type column has "job" in it.
query = """SELECT case_number, primary_type
            FROM `bigquery-public-data.chicago_crime.crime`
            WHERE primary_type = "THEFT" """

# check how big this query will be
chicago_crime.estimate_query_size(query)
# only run this query if it's less than 135 MB
theft_cases = chicago_crime.query_to_pandas_safe(query, max_gb_scanned=0.135)
theft_cases
# count the number of theft cases.
theft_cases.case_number.count()