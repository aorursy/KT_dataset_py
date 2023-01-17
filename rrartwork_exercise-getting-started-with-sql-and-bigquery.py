# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *





# create a helper object for our bigquery dataset

import bq_helper

chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                         dataset_name = "chicago_crime")

print("Setup Complete")
 # Write the code you need here to figure out the answer

chicago_crime.list_tables()
num_tables = 1  # store the answer as num_tables and then run this cell



q_1.check()
q_1.hint()

q_1.solution()
chicago_crime.table_schema('crime')
chicago_crime.head('crime')
chicago_crime.head('crime', selected_columns="primary_type", num_rows=10)
# Write the code to figure out the answer

chicago_crime.table_schema('crime')
num_timestamp_fields = 2 # put your answer here



q_2.check()
q_2.hint()

q_2.solution()
# Write the code here to explore the data so you can find the answer

chicago_crime.head('crime')
fields_for_plotting = ['latitude', 'longitude']



q_3.check()
q_3.hint()

q_3.solution()
# Scratch space for your code
from google.cloud import bigquery
# Create a "Client" object

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

basketball_ref = client.dataset("ncaa_basketball", project="bigquery-public-data")



# API request - fetch the dataset

bballdataset = client.get_dataset(basketball_ref)
# List all the tables in the "hacker_news" dataset

ncaatables = list(client.list_tables(bballdataset))



# Print names of all tables in the dataset (there are four!)

for table in ncaatables:  

    print(table.table_id)
# Construct a reference to the "full" table

ncaatable_ref = basketball_ref.table("mbb_players_games_sr")



# API request - fetch the table

playerstable = client.get_table(ncaatable_ref)
# Print information on all the columns in the "full" table in the "hacker_news" dataset

playerstable.schema
# Preview the first 20 lines of the "full" table

client.list_rows(playerstable, max_results=20).to_dataframe()
# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)
# Construct a reference to the "full" table

table_ref = dataset_ref.table("full")



# API request - fetch the table

table = client.get_table(table_ref)

# Print information on all the columns in the "full" table in the "hacker_news" dataset

table.schema
# Preview the first 20 lines of the "full" table

client.list_rows(table, max_results=20).to_dataframe()
# Preview the first five entries in the "by" column of the "full" table

client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()
comment_ref = dataset_ref.table("comments")



#fetch data

comment_table = client.get_table(comment_ref)
comment_table.schema
# Preview the first 20 lines of the "full" table

client.list_rows(comment_table, max_results=20).to_dataframe()
stories_ref = dataset_ref.table("stories")



#fetch data

stories_table = client.get_table(stories_ref)
stories_table.schema
# Preview the first 20 lines of the "full" table

client.list_rows(stories_table, max_results=20).to_dataframe()