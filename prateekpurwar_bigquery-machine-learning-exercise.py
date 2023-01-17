# Set your own project id here

PROJECT_ID = 'bqml-250120' # a string, like 'kaggle-bigquery-240818'



from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID, location="US")

dataset = client.create_dataset('model_dataset', exists_ok=True)



from google.cloud.bigquery import magics

from kaggle.gcp import KaggleKernelCredentials

magics.context.credentials = KaggleKernelCredentials()

magics.context.project = PROJECT_ID
# create a reference to our table

table = client.get_table("bigquery-public-data.austin_bikeshare.bikeshare_trips")



# look at five rows from our dataset

client.list_rows(table, max_results=5).to_dataframe()
%load_ext google.cloud.bigquery
# we can use cross validation method 
%%bigquery dataframe_name
%%bigquery
%%bigquery
## Thought question answer here
%%bigquery
%%bigquery
## Thought question answer here