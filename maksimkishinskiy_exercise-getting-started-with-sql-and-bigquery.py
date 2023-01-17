# Set up feedack system
from learntools.core import binder
binder.bind(globals())
from learntools.sql.ex1 import *
print("Setup Complete")
from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "chicago_crime" dataset
dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)
# Write the code you need here to figure out the answer
tables = list(client.list_tables(dataset))
num_tables = len(tables) # Store the answer as num_tables and then run this cell
for table in tables:
    print(table.table_id)
q_1.check()
q_1.hint()
q_1.solution()
# Write the code to figure out the answer
table_ref = dataset_ref.table("crime") # Construct a reference to the "full" table
table = client.get_table(table_ref) # API request - fetch the table
table.schema
# Preview the first five lines of the "full" table
client.list_rows(table, selected_fields = table.schema[:10], max_results=10).to_dataframe()
num_timestamp_fields = 2 # Put your answer here

q_2.check()
q_2.hint()
q_2.solution()
# Write the code here to explore the data so you can find the answer
table_ref = dataset_ref.table("crime") # Construct a reference to the "full" table
table = client.get_table(table_ref) # API request - fetch the table
table.schema
fields_for_plotting = ['latitude', 'longitude'] # Put your answers here
fields_for_plotting = ['x_coordinate', 'y_coordinate'] # Second variant

q_3.check()
q_3.hint()
q_3.solution()
# Scratch space for your code

# Query to get the score column from every row where the type column has value "job"
query = """
        SELECT date, location_description
        FROM `bigquery-public-data.chicago_crime.crime`
        WHERE primary_type = "NARCOTICS" 
        """

# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)

# API request - dry run query to estimate costs
dry_run_query_narcotics = client.query(query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_narcotics.total_bytes_processed))
