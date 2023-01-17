# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")
from google.cloud import bigquery



client = bigquery.Client()



dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))

c = 0 

for i in tables:

    c=c+1
num_tables = c  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
table_ref = dataset_ref.table("crime")



crime_table = client.get_table(table_ref)



d = 0

for i in range(len(crime_table.schema)):

    if crime_table.schema[i].field_type=='TIMESTAMP':

        d=d+1
num_timestamp_fields = d # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
client.list_rows(crime_table,max_results=10).to_dataframe()
fields_for_plotting = ['latitude', 'longitude'] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
client.list_rows(crime_table,max_results=5).to_dataframe()