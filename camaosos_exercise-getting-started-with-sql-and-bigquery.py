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



print(num_tables)

for table in tables:  

    print(table.table_id)



q_1.check()
q_1.hint()

q_1.solution()
# Write the code to figure out the answer

crime_table = client.get_table(dataset_ref.table("crime"))

print(crime_table.schema)
#import pandas



#query = """SELECT date FROM `bigquery-public-data.chicago_crime.crime` WHERE year=2010"""

#query2 = """SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME =`bigquery-public-data.chicago_crime.crime`"""

#query_job = client.query(query)

#query2_job = client.query(query2)



#print(query_job.to_dataframe())



timestamp_fields = [field for field in crime_table.schema if (field.field_type == 'TIMESTAMP')]

print(timestamp_fields)



num_timestamp_fields = len(timestamp_fields) # Put your answer here



q_2.check()
q_2.hint()

q_2.solution()
# Write the code here to explore the data so you can find the answer



query_latitude = """SELECT latitude FROM `bigquery-public-data.chicago_crime.crime`"""

query_longitude = """SELECT longitude FROM `bigquery-public-data.chicago_crime.crime`"""



#query_latitude_job = client.query(query_latitude)

#query_longitude_job = client.query(query_longitude)



#crime_table.latitude



client.list_rows(crime_table, max_results=5).to_dataframe()



#print(query_latitude_job.to_dataframe())
fields_for_plotting = ["longitude", "latitude"] # Put your answers here

#fields_for_plotting2 = [query_longitude_job.to_dataframe().values, query_latitude_job.to_dataframe().values] # Put your answers here



q_3.check()
q_3.hint()

q_3.solution()
# Scratch space for your code