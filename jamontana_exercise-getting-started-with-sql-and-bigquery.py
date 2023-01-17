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
tables = list(client.list_tables(dataset))



for table in tables:  

    print(table.table_id)

    

print(len(tables))
num_tables = 1  # Store the answer as num_tables and then run this cell



q_1.check()
q_1.hint()

q_1.solution()
table_ref = dataset_ref.table("crime")

table = client.get_table(table_ref)

aux=0

for i in table.schema:

    algo=str(i).split(",")[1]

    if algo ==" 'TIMESTAMP'":

        aux+=1

print("there are {} columns with TIMESTAMP".format(aux))
num_timestamp_fields = 2 # Put your answer here



q_2.check()
q_2.hint()

q_2.solution()
client.list_rows(table, max_results=5).to_dataframe()
#fields_for_plotting = ["latitude", "longitude"]

#or

#fields_for_plotting_2 = ["location"]
fields_for_plotting = ["latitude", "longitude"] # Put your answers here



q_3.check()
q_3.hint()

q_3.solution()
import matplotlib.pylab as plt

client.list_rows(table, max_results=500).to_dataframe()[fields_for_plotting].plot(x="latitude",y="longitude",kind="scatter")
data=client.list_rows(table, max_results=500).to_dataframe()
data.plot(y="x_coordinate",x="y_coordinate",kind="scatter")