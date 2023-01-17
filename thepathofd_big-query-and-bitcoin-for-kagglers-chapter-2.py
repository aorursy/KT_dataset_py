import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from google.cloud import bigquery
import squarify
import operator
import time
import re
import gc
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

class MyBQHelper(object):
    """
    Helper class to simplify common BigQuery tasks like executing queries,
    showing table schemas, etc without worrying about table or dataset pointers.
    See the BigQuery docs for details of the steps this class lets you skip:
    https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html
    """

    BYTES_PER_GB = 2**30

    def __init__(self, active_project, dataset_name, max_wait_seconds=180):
        self.project_name = active_project
        self.dataset_name = dataset_name
        self.max_wait_seconds = max_wait_seconds
        self.client = bigquery.Client()
        self.__dataset_ref = self.client.dataset(self.dataset_name, project=self.project_name)
        self.dataset = None
        self.tables = dict()  # {table name (str): table object}
        self.__table_refs = dict()  # {table name (str): table reference}
        self.total_gb_used_net_cache = 0

    def __fetch_dataset(self):
        # Lazy loading of dataset. For example,
        # if the user only calls `self.query_to_pandas` then the
        # dataset never has to be fetched.
        if self.dataset is None:
            self.dataset = self.client.get_dataset(self.__dataset_ref)

    def __fetch_table(self, table_name):
        # Lazy loading of table
        self.__fetch_dataset()
        if table_name not in self.__table_refs:
            self.__table_refs[table_name] = self.dataset.table(table_name)
        if table_name not in self.tables:
            self.tables[table_name] = self.client.get_table(self.__table_refs[table_name])

    def table_schema(self, table_name):
        """
        Get the schema for a specific table from a dataset
        """
        self.__fetch_table(table_name)
        return(self.tables[table_name].schema)

    def list_tables(self):
        """
        List the names of the tables in a dataset
        """
        self.__fetch_dataset()
        return([x.table_id for x in self.client.list_tables(self.dataset)])

    def estimate_query_size(self, query):
        """
        Estimate gigabytes scanned by query.
        Does not consider if there is a cached query table.
        See https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
        """
        my_job_config = bigquery.job.QueryJobConfig()
        my_job_config.dry_run = True
        my_job = self.client.query(query, job_config=my_job_config)
        return my_job.total_bytes_processed / self.BYTES_PER_GB

    def query_to_pandas(self, query):
        """
        Take a SQL query & return a pandas dataframe
        """
        my_job = self.client.query(query)
        start_time = time.time()
        while not my_job.done():
            if (time.time() - start_time) > self.max_wait_seconds:
                print("Max wait time elapsed, query cancelled.")
                self.client.cancel_job(my_job.job_id)
                return None
            time.sleep(0.1)
        #print(my_job._properties.get('status'))
        if my_job.total_bytes_billed:
            self.total_gb_used_net_cache += my_job.total_bytes_billed / self.BYTES_PER_GB
        return my_job.to_dataframe()

    def query_to_pandas_safe(self, query, max_gb_scanned=1):
        """
        Execute a query if it's smaller than a certain number of gigabytes
        """
        query_size = self.estimate_query_size(query)
        if query_size <= max_gb_scanned:
            return self.query_to_pandas(query)
        msg = "Query cancelled; estimated size of {0} exceeds limit of {1} GB"
        print(msg.format(query_size, max_gb_scanned))

    def head(self, table_name, num_rows=5, start_index=None, selected_columns=None):
        """
        Get the first n rows of a table as a DataFrame
        """
        self.__fetch_table(table_name)
        active_table = self.tables[table_name]
        schema_subset = None
        if selected_columns:
            schema_subset = [col for col in active_table.schema if col.name in selected_columns]
        results = self.client.list_rows(active_table, selected_fields=schema_subset,
            max_results=num_rows, start_index=start_index)
        results = [x for x in results]
        return pd.DataFrame(
            data=[list(x.values()) for x in results], columns=list(results[0].keys()))
    
    
from base58check import b58encode
from hashlib import sha256
from hashlib import new as hnew

def pubkey_to_hash(pubkey_string):
    hash_160=hnew('ripemd160')
    hash_160.update(sha256(bytes.fromhex(pubkey_string)).digest())
    return hash_160.hexdigest()

def address_from_hash(hash_string,pubkey=True):
    prefix='00' if pubkey else '05'
    PubKeyHash=bytes.fromhex(prefix+hash_string)
    checksum=sha256(sha256(PubKeyHash).digest()).digest()[:4]
    haa=PubKeyHash+checksum
    return b58encode(haa).decode('utf-8')

def address_from_pubkey(pubkey_string,pubkey=True):
    return address_from_hash(pubkey_to_hash(pubkey_string),pubkey)
    
bq_assistant = MyBQHelper('bigquery-public-data','bitcoin_blockchain',max_wait_seconds=6000)
QUERY='''
#standardSQL
SELECT TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day, COUNT(DISTINCT transaction_id) as volume
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
WHERE timestamp<1529363632000
GROUP BY day
'''
daily_volume=bq_assistant.query_to_pandas(QUERY)
daily_volume.set_index('day').volume.plot(title='daily volume of transactions',figsize=(20,6));
QUERY='''
#standardSQL
WITH transaction_traffic AS (
                     (SELECT TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day, 'input' as traffic, COUNT(DISTINCT i.input_pubkey_base58) AS count
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(inputs) AS i
                      WHERE (i.input_pubkey_base58 IS NOT NULL) AND (timestamp<1529363632000)
                      GROUP BY day,transaction_id)
                     UNION ALL
                     (SELECT TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day, 'output' as traffic, COUNT(DISTINCT o.output_pubkey_base58) AS count
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(outputs) AS o
                      WHERE (o.output_pubkey_base58 IS NOT NULL)  AND (timestamp<1529363632000)
                      GROUP BY day, transaction_id)
                      )
SELECT day, traffic, AVG(count) as average_count,MAX(count) as max_count,MIN(count) as min_count
FROM transaction_traffic
GROUP BY day,traffic
'''
daily_diversity=bq_assistant.query_to_pandas(QUERY)
fig,axes=plt.subplots(nrows=3,figsize=(20,24))
daily_diversity.set_index(['day','traffic'])['average_count'].unstack(level=1).fillna(0).plot(title='average daily tansaction diversity',ax=axes[0]);
daily_diversity.set_index(['day','traffic']).max_count.unstack(level=1).fillna(0).plot(title='maximum daily tansaction diversity',ax=axes[1]);
daily_volume.set_index('day').volume.plot(title='daily volume of transactions',ax=axes[2]);
#daily_diversity.set_index(['day','traffic']).min_count.unstack(level=1).fillna(0).plot(title='minimum daily tansaction diversity',ax=axes[2]);
#time TIMESTAMP_MILLIS(timestamp)
QUERY='''
#standardSQL
WITH transaction_table AS (
                     (SELECT transaction_id, i.input_pubkey_base58 AS address
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(inputs) AS i
                      WHERE timestamp<1529363632000)
        UNION ALL
                     (SELECT transaction_id, o.output_pubkey_base58 AS address
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(outputs) AS o
                      WHERE timestamp<1529363632000)
                      ),
     volume_table AS (
                      SELECT address, COUNT(DISTINCT transaction_id) as volume
                      FROM transaction_table
                      GROUP BY address)
SELECT volume, count(address) AS num_addresses
FROM volume_table
GROUP BY volume
'''

volume=bq_assistant.query_to_pandas(QUERY).sort_values('volume').drop(0).set_index('volume')
plt.figure(figsize=(20,6))
squarify.plot(sizes=volume.num_addresses,label=volume.index);
QUERY='''
#standardSQL
WITH transaction_table AS (
                     (SELECT transaction_id, i.input_pubkey_base58 AS address, -1 as traffic
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(inputs) AS i
                      WHERE timestamp<1529363632000)
                     UNION ALL
                     (SELECT transaction_id, o.output_pubkey_base58 AS address, 1 as traffic
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(outputs) AS o
                      WHERE timestamp<1529363632000)
                      ),
     volume_table AS (
                      SELECT address, COUNT(DISTINCT transaction_id) as volume, 
                        CASE AVG(DISTINCT traffic) 
                          WHEN -1 THEN 'outgoing funds only'
                          WHEN 0 THEN 'both'
                          WHEN 1 THEN 'incoming funds only'
                         END AS transaction_types
                      FROM transaction_table
                      GROUP BY address
                      HAVING volume <3)
SELECT volume, transaction_types, count(address) AS num_addresses
FROM volume_table
GROUP BY volume,transaction_types
'''
type_by_volume=bq_assistant.query_to_pandas(QUERY).sort_values('num_addresses',ascending=False)
type_by_volume
single_trans=type_by_volume.loc[lambda x: x.volume==1]
double_trans=type_by_volume.loc[lambda x: x.volume==2]
fig,axes = plt.subplots(ncols=2,figsize=(20,6))
squarify.plot(sizes=np.log(single_trans.num_addresses.values),label=single_trans.transaction_types,ax=axes[0]);
axes[0].set_title('Log-log distribution of addresses with 1 transaction')
squarify.plot(sizes=np.log(double_trans.num_addresses.values),label=double_trans.transaction_types,ax=axes[1]);
axes[1].set_title('Log-log distribution of addresses with 2 transactions');
QUERY='''
#standardSQL
WITH transaction_table AS (
                     (SELECT transaction_id, i.input_pubkey_base58 AS address, -1 as traffic
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(inputs) AS i 
                       WHERE timestamp<1526668467000)
                     UNION ALL
                     (SELECT transaction_id, o.output_pubkey_base58 AS address, 1 as traffic
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(outputs) AS o 
                       WHERE timestamp<1526668467000)
                      )
SELECT transaction_id, i.input_pubkey_base58 AS address, i.input_script_string as script
FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(inputs) AS i 
WHERE i.input_pubkey_base58 IN (SELECT address
                      FROM transaction_table
                      GROUP BY address
                      HAVING (COUNT(DISTINCT transaction_id) <3) AND (AVG(DISTINCT traffic) = -1))
'''
out_only=bq_assistant.query_to_pandas(QUERY)
out_only.info()
out_only.head()
n=1
out_only.iloc[n,0]
x=out_only.iloc[n,2]
x
x=re.search(r'PUSHDATA\(3\d\)\[(\w+)\]',x).group(1)
print('The address in our table is {}, \nwhile the script is a P2SH script with addres {}'.format(address_from_pubkey(x),address_from_pubkey(x,False)))
print('This script is actually a P2PK, and the first and last bytes of the 35 bytes script are PUSHDATA(33) and CHECKSIG, respectively 21 and ac.')
print('The actuall public key used is {}, which curiously has not been used for any transaction'.format(address_from_pubkey(x[2:-2])))

QUERY='''#standardSQL
WITH transaction_table AS (
                     (SELECT transaction_id, i.input_pubkey_base58 AS address, -1 as traffic, 0 as BTC
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(inputs) AS i )
                     UNION ALL
                     (SELECT transaction_id, o.output_pubkey_base58 AS address, 1 as traffic, o.output_satoshis/100000000.0 as BTC
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`, UNNEST(outputs) AS o )
                      )
SELECT address, SUM(BTC) AS amount
FROM transaction_table
GROUP BY address
HAVING COUNT(traffic)=SUN(traffic)
'''