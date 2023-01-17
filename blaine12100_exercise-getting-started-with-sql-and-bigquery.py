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



'''First we need to get the list of tables in a database using the client.list_tables method.Then for each table 

in the client list,we can get the relavent details using a list comprehension since the list table returns us a 

iterator object.'''



list_tables=list(client.list_tables(dataset))



table_names=[table.table_id for table in list_tables]

    

print(len(table_names))
num_tables = 1  # Store the answer as num_tables and then run this cell



q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the answer



'''Create a reference to a particular table.This is done so that we know that we need to use this 

particular table only.Using the dataset_ref allows us to get the table for a particular Database.After getting 

the reference,fetch the table for which we have created a reference.Then check the schema of the Table.



The schema refers to the structure of the dataabse itself.It gives us information about how the columns are 

named,what data type do they accept,whether the field allows null values etc.

'''



table_reference=dataset_ref.table("crime")



table_schema=client.get_table(table_reference)



table_schema.schema
num_timestamp_fields = 2 # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer



print(client.list_rows(table_schema,max_results=10).to_dataframe())
'''There are multiple fields that we can use.We can also use the x and y coordinates in this case as they would

refer to the same coordinates if plotted on a map.'''



fields_for_plotting = ["latitude", "longitude"] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code

'''This bigquery synatx is a little confusing for me to undetstand.Like the column indices always need to be 

in a range and if you try to show more than 3 columns at a time,the rows are rendered as dots which 

required me to write a separate query for the same'''





print(client.list_rows(table_schema,selected_fields=table_schema.schema[19:21],max_results=10).to_dataframe())



print(client.list_rows(table_schema,selected_fields=table_schema.schema[21:],max_results=10).to_dataframe())