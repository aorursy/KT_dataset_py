# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *





# create a helper object for our bigquery dataset

import bq_helper

chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                         dataset_name = "chicago_crime")

print("Setup Complete")
#For reference, here is all the code you saw in the first tutorial:



'''

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")

hacker_news.list_tables()

hacker_news.table_schema("full")

hacker_news.head("full")

hacker_news.head("full", selected_columns="by", num_rows=10)

'''
chicago_crime.list_tables()
num_tables = len(chicago_crime.list_tables())  # store the answer as num_tables and then run this cell



q_1.check()
# q_1.hint()

# q_1.solution()
chicago_crime.table_schema("crime")
num_timestamp_fields = sum(chicago_crime.table_schema("crime").type == "TIMESTAMP")



q_2.check()
# q_2.hint()

# q_2.solution()
chicago_crime.table_schema("crime").name[[15,16,19,20]]
fields_for_plotting = ['latitude', 'longitude']



q_3.check()
# q_3.hint()

# q_3.solution()
chicago_crime.head("crime")