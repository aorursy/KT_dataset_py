# Put your own project id here

PROJECT_ID = 'kaggle-bq-test-248917'



# create a client instance for your project

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID, location="US")
# create small array & save it as a csv

# (This is too small to warrent using BigQuery, it's

# just here as an example)

import numpy

a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])

numpy.savetxt("foo.csv", a, delimiter=",")
# create a new datset

client.create_dataset("new_dataset")
# create a new table in that dataset ()

client.create_table(f"{PROJECT_ID}.new_dataset.new_table")
# some variables

filename = 'foo.csv' # this is the file path to your csv

dataset_id = 'new_dataset'

table_id = 'new_table'



# tell the client everything it needs to know to upload our csv

dataset_ref = client.dataset(dataset_id)

table_ref = dataset_ref.table(table_id)

job_config = bigquery.LoadJobConfig()

job_config.source_format = bigquery.SourceFormat.CSV

job_config.autodetect = True



# load the csv into bigquery

with open(filename, "rb") as source_file:

    job = client.load_table_from_file(source_file, table_ref, job_config=job_config)



job.result()  # Waits for table load to complete.



# looks like everything worked :)

print("Loaded {} rows into {}:{}.".format(job.output_rows, dataset_id, table_id))
# query (you won't want to use SELECT * unless your dataset is very small)

query = f""" SELECT * 

        FROM `{PROJECT_ID}.new_dataset.new_table`"""



# Set up the query

query_job = client.query(query)



# API request - run the query, and return a pandas DataFrame

data = query_job.to_dataframe()
# check out the data we got back :)

data