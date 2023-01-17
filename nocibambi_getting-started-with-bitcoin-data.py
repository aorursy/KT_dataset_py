from google.cloud import bigquery

client = bigquery.Client()
def get_atts(obj, filter=""):

    """Helper function wich prints the public attributes and methods of the given object.

    Can filter the results for simple term.

    - `obj`: the Python object of interest

    - `filter`, str: The filter term

    """

    return [a for a in dir(obj) if not a.startswith('_') and filter in a]
get_atts(client)
get_atts(client, 'list')
project = 'bigquery-public-data'

dataset = 'crypto_bitcoin'
client.list_tables(f'{project}.{dataset}')
for ds in client.list_tables(f'{project}.{dataset}'):

    print(get_atts(ds))

    break
[ds.table_id for ds in client.list_tables(f'{project}.{dataset}')]
get_atts(client, 'table')
table = 'transactions'

trans_ref = client.get_table(f'{project}.{dataset}.{table}')

get_atts(trans_ref)
trans_ref.schema
import pandas as pd



blocks_firstrows_df = pd.DataFrame(

    [

        dict(row) for row

        in client.list_rows(trans_ref, max_results=5)

    ]

)

blocks_firstrows_df
blocks_firstrows_df['outputs'][0]
query = """

    SELECT

        `hash`,

        block_timestamp,

        input_count,

        input_value,

        output_count,

        output_value

    FROM `bigquery-public-data.crypto_bitcoin.transactions`

    WHERE

        EXTRACT(YEAR FROM block_timestamp) = 2017 AND

        EXTRACT(MONTH FROM block_timestamp) = 09    

"""
def estimate_gigabytes_scanned(query, bq_client):

    """A useful function to estimate query size. 

    Originally from here: https://www.kaggle.com/sohier/beyond-queries-exploring-the-bigquery-api/

    """

    # We initiate a `QueryJobConfig` object

    # API description: https://googleapis.dev/python/bigquery/latest/generated/google.cloud.bigquery.job.QueryJobConfig.html

    my_job_config = bigquery.job.QueryJobConfig()

    

    # We turn on 'dry run', by setting the `QueryJobConfig` object's `dry_run` attribute.

    # This means that we do not actually run the query, but estimate its running cost. 

    my_job_config.dry_run = True



    # We activate the job_config by passing the `QueryJobConfig` to the client's `query` method.

    my_job = bq_client.query(query, job_config=my_job_config)

    

    # The results comes as bytes which we convert into Gigabytes for better readability

    BYTES_PER_GB = 2**30

    estimate = my_job.total_bytes_processed / BYTES_PER_GB

    

    print(f"This query will process {estimate} GBs.")
query = """

    SELECT *

    FROM `bigquery-public-data.crypto_bitcoin.transactions`

"""
estimate_gigabytes_scanned(query, client)
query = """

    SELECT

        `hash`,

        block_timestamp,

        input_count,

        input_value,

        output_count,

        output_value

    FROM `bigquery-public-data.crypto_bitcoin.transactions`

    WHERE

        EXTRACT(YEAR FROM block_timestamp) = 2017 AND

        EXTRACT(MONTH FROM block_timestamp) = 09    

"""
estimate_gigabytes_scanned(query, client)
bytes_in_gigabytes = 2**30



safe_config = bigquery.QueryJobConfig(

    maximum_bytes_billed=54 * bytes_in_gigabytes

)



query_job = client.query(query, job_config=safe_config)
from time import time

start = time()
result = query_job.result()

df = result.to_dataframe()
duration = (time() - start) / 60

print(f"Elapsed time: {duration:.2f} minutes")
df.shape
df.head()
mem_use = df.memory_usage(deep=True)

print(mem_use)



mem_use.sum() / bytes_in_gigabytes
df.to_csv('transactions_201709.csv', index=False)
df = pd.read_csv('transactions_201709.csv')
df
df['block_timestamp'] = pd.to_datetime(df['block_timestamp'] )
df['avg_out_val'] = df['output_value'] / df['output_count']

df['avg_in_val'] = df['input_value'] / df['input_count']
df
df.set_index('block_timestamp')['output_value'].plot(figsize=(18, 8))
df.set_index('block_timestamp')['input_value'].plot(figsize=(18, 8), color='orange')
df.set_index('block_timestamp')['avg_out_val'].plot(figsize=(18, 8))
df.set_index('block_timestamp')['avg_in_val'].plot(figsize=(18, 8), color='orange')
df.groupby(df['block_timestamp'].dt.date)['output_value'].sum().plot(kind='bar', figsize=(18, 8))
df.groupby(df['block_timestamp'].dt.date)['input_value'].sum().plot(kind='bar', figsize=(18, 8))
df.groupby(df['block_timestamp'].dt.date)['avg_out_val'].sum().plot(kind='bar', figsize=(18, 8))
df.groupby(df['block_timestamp'].dt.date)['avg_in_val'].sum().plot(kind='bar', figsize=(18, 8))
(

    df

    .groupby(df['block_timestamp'].dt.weekday)

    ['output_value'].sum()

    .rename(index={i: day for i, day in enumerate(['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'])})

    .plot(kind='bar', figsize=(18, 8))

)
(

    df

    .groupby(df['block_timestamp'].dt.weekday)

    ['input_value'].sum()

    .rename(index={i: day for i, day in enumerate(['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'])})

    .plot(kind='bar', figsize=(18, 8))

)
(

    df

    .groupby(df['block_timestamp'].dt.weekday)

    ['avg_out_val'].sum()

    .rename(index={i: day for i, day in enumerate(['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'])})

    .plot(kind='bar', figsize=(18, 8))

)
(

    df

    .groupby(df['block_timestamp'].dt.weekday)

    ['avg_in_val'].sum()

    .rename(index={i: day for i, day in enumerate(['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'])})

    .plot(kind='bar', figsize=(18, 8))

)
df.groupby(df['block_timestamp'].dt.hour)['output_value'].sum().plot(kind='bar', figsize=(18, 8))