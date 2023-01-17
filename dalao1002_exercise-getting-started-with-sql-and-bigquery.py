# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *





# create a helper object for our bigquery dataset

import bq_helper

chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                         dataset_name = "chicago_crime")

print("Setup Complete")
len(chicago_crime.list_tables()) # Write the code you need here to figure out the answer
num_tables = 1  # store the answer as num_tables and then run this cell



q_1.check()
q_1.hint()

q_1.solution()
chicago_crime.table_schema("crime") # Write the code to figure out the answer

num_timestamp_fields = 2 # put your answer here



q_2.check()
q_2.hint()

q_2.solution()
chicago_crime.table_schema('crime') # Write the code here to explore the data so you can find the answer
fields_for_plotting = ["latitude", "longitude"]



q_3.check()
q_3.hint()

q_3.solution()
chicago_crime.head("crime") # Scratch space for your code