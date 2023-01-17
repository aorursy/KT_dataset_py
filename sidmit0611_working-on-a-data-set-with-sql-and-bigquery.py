# importing the bigquery library from google.cloud
from google.cloud import bigquery
# creating a client object
# remeber this client object will help us in fetching the data
client = bigquery.Client()
# now that the client in initialised
# lets import the data from 'bigquery-public-data' project 
# we will be using openaq dataset to perform some sql operations
# remeber that 'bigquery-public-data' is the project under which 'openaq' (data regarding air quality) dataset is present
dataset_ref = client.dataset('openaq', project = 'bigquery-public-data')
# taking the table reference in our variable
dataset = client.get_dataset(dataset_ref)
# now we will see the tables in our data
tables = list(client.list_tables(dataset))

for i in tables:
    print(i.table_id)
# so we have one table present in our 'openaq' database
# now we want to fetch the information of the table 'global_air_quality'
# for that we will create the table reference and then we will fetch the information from that reference into our variable

table_ref = dataset_ref.table('global_air_quality')

table = client.get_table(table_ref)
table.schema
# printing the result in table 'global_air_quality'

client.list_rows(table, max_results = 5).to_dataframe()
# lets suppose we want to fecth the value column from 'global_air_quality' table of 'openaq' database which is present in 
# 'bigquery-public-data'

query = """ select value from `bigquery-public-data.openaq.global_air_quality` where city = 'Bengaluru' """
# asking our client to take the query
query_job = client.query(query)
# storing the result into our variable
values = query_job.to_dataframe()
values.head()
# here we can see that how we can use our pandas function along with the sql
values_greater_than_50 = values[values['value'] >= 50]
values_greater_than_50
query = """ select city from `bigquery-public-data.openaq.global_air_quality` where country = 'IN' """
query_job = client.query(query)
cities = query_job.to_dataframe()
cities['city'].value_counts().head(10)
query = """select * from `bigquery-public-data.openaq.global_air_quality` """
# submitting the query to the dataset
query_job = client.query(query)
df = query_job.to_dataframe()
df.shape
df.head()
df.describe()
