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
# this query looks in the full table in the chicago_crime
# dataset, then gets the primary_type,community_area column from every row where 
# the type column has "arrest=True" in it.
query = """SELECT primary_type,district,ward,community_area
            FROM `bigquery-public-data.chicago_crime.crime`
            WHERE arrest = True"""

# check how big this query will be
chicago_crime.estimate_query_size(query)
# only run this query if it's less than 100 MB
chicago_crime.query_to_pandas_safe(query, max_gb_scanned=0.1)
#And here's an example where the same query returns a dataframe.

# check out the scores of job postings (if the 
# query is smaller than 1 gig)
chicago_crime_DataFrame = chicago_crime.query_to_pandas_safe(query)
chicago_crime_DataFrame.head()
