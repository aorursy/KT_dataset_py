# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *





# create a helper object for our bigquery dataset

import bq_helper

chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                         dataset_name = "chicago_crime")

print("Setup Complete")
chicago_crime.list_tables()
num_tables = 1



q_1.check()
# q_1.hint()

# q_1.solution()
chicago_crime.table_schema("crime")

num_timestamp_fields = 2



q_2.check()
# q_2.hint()

# q_2.solution()
chicago_crime.head("crime")
fields_for_plotting = ["x_coordinate", "y_coordinate"]



q_3.check()
# q_3.hint()

# q_3.solution()
chicago_crime.head("crime")