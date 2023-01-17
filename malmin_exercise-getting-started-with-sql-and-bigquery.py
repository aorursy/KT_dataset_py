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
num_tables = len(tables)  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the answer

for table in tables:  

    print(table.table_id)

table_ref = dataset_ref.table("crime")

# API request - fetch the table

table = client.get_table(table_ref)

tsfields = [field for field in table.schema if field.field_type=='TIMESTAMP']

print(tsfields)
num_timestamp_fields = len(tsfields) # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer

table.schema
fields_for_plotting = ['x_coordinate', 'y_coordinate'] # Put your answers here



q_3.check()
q_3.hint()

#q_3.solution()
# Scratch space for your code

client.list_rows(table, max_results=5).to_dataframe()

# Only run the query if it's less than 1 GB

ONE_GB = 1000*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_GB)



query = """

        SELECT primary_type,description

        FROM `bigquery-public-data.chicago_crime.crime`

        WHERE arrest = True

        """



def safeQuery(query):

    # Create a "Client" object

    client = bigquery.Client()

    # Set up the query

    query_job = client.query(query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame

    df = query_job.to_dataframe()

    return df



def checkSize(query):

    # Create a QueryJobConfig object to estimate size of query without running it

    dry_run_config = bigquery.QueryJobConfig(dry_run=True)

    # API request - dry run query to estimate costs

    dry_run_query_job = client.query(query, job_config=dry_run_config)

    return dry_run_query_job.total_bytes_processed





size = checkSize(query)

print("This query will process {} bytes.".format(size))

if size < ONE_GB and df.empty:

    df = safeQuery(query)

df.shape
df.dtypes
df.describe()
df.groupby('primary_type').size().sort_values(ascending=False)