# Set your own project id here

PROJECT_ID = "kaggle-bqml-mimi" # a string, like 'kaggle-bigquery-240818'



  

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID, location="US")

#from google.cloud import bigquery

#client = bigquery.Client(project=PROJECT_ID, location="US")

dataset = client.create_dataset('model_dataset', exists_ok=True)



from google.cloud.bigquery import magics

from kaggle.gcp import KaggleKernelCredentials

magics.context.credentials = KaggleKernelCredentials()

magics.context.project = PROJECT_ID
%load_ext google.cloud.bigquery
# You can write your notes here
%%bigquery dataframe_name
%%bigquery
%%bigquery
## Thought question answer here
%%bigquery
%%bigquery
## Thought question answer here