from google.cloud import bigquery
import pandas as pd
from bq_helper import BigQueryHelper
import numpy as np 
import pandas as pd 
# bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")
bitcoin_blockchain = BigQueryHelper("bigquery-public-data","bitcoin_blockchain")
%%time
bitcoin_blockchain.list_tables()
client= bigquery.Client()
github_dset = client.get_dataset(client.dataset('bitcoin_blockchain', project='bigquery-public-data'))
blocks_table = client.get_table(github_dset.table('blocks'))
BYTES_PER_GB = 2**30
print(f'The blocks table is {int(blocks_table.num_bytes/BYTES_PER_GB)} GB')
client= bigquery.Client()
github_dset = client.get_dataset(client.dataset('bitcoin_blockchain', project='bigquery-public-data'))
transactions_table = client.get_table(github_dset.table('transactions'))
BYTES_PER_GB = 2**30
print(f'The transactions table is {int(transactions_table.num_bytes/BYTES_PER_GB)} GB')
%%time
bitcoin_blockchain.table_schema("blocks")
%%time
bitcoin_blockchain.head("blocks")
#test_data=bitcoin_blockchain.head("transactions")
%%time
bitcoin_blockchain.table_schema("transactions")
%%time
bitcoin_blockchain.head("transactions")
sample_data=bitcoin_blockchain.head("transactions")
# x["input_pubkey_base58"]="1L6eTM7CdfU9eQwYWf3u8mFPrfxsiqV8hy"
x=sample_data.iloc[2].inputs[0]
print(x)
x=sample_data.iloc[3].outputs[0]
x["output_pubkey_base58"]= "1NhG34WuLN6aWEdbzt6VE4tJQLUem8J43Z"
print(x)
print(sample_data.iloc[2].outputs[0])
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

# Query by Allen Day, GooglCloud Developer Advocate (https://medium.com/@allenday)
query = """
#standardSQL
SELECT
  o.day,
  COUNT(DISTINCT(o.output_key)) AS recipients
FROM (
  SELECT
    TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day,
    output.output_pubkey_base58 AS output_key
  FROM `bigquery-public-data.bitcoin_blockchain.transactions`,
    UNNEST(outputs) AS output ) AS o
GROUP BY day
ORDER BY day
"""

query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
transactions.head(10)
transactions_total = bitcoin_blockchain.query_to_pandas_safe( """
       WITH time AS (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month ,day
            ORDER BY year, month,day
        """, max_gb_scanned=23)

transactions_total.head(10)
import matplotlib.pyplot as plt
%matplotlib inline
# plot monthly bitcoin transactions
plt.rcParams["figure.figsize"] = (20,8)
plt.plot(transactions_total)
plt.title("Bitcoin Transcations")
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe( """
       WITH time AS (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """, max_gb_scanned=23)

transactions_per_month 
#import plotting library
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = (20,8)
# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
transaction_by_day_2017 = bitcoin_blockchain.query_to_pandas_safe("""
     WITH time AS (
           SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
           transaction_id
       FROM `bigquery-public-data.bitcoin_blockchain.transactions`
     )
    SELECT COUNT(transaction_id) AS number,
       EXTRACT(month FROM trans_time) AS month,
       EXTRACT(day FROM trans_time) AS day
    FROM time
    WHERE EXTRACT(year FROM trans_time)=2017
    GROUP BY month,day
    ORDER BY month,day
""",max_gb_scanned=23)
transaction_by_day_2017
plt.rcParams["figure.figsize"] = (20,8)
plt.plot(transaction_by_day_2017.number)
plt.title('Daily Number of Transcations in 2017')
transaction_by_day_2018 = bitcoin_blockchain.query_to_pandas_safe("""
     WITH time AS (
           SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
           transaction_id
       FROM `bigquery-public-data.bitcoin_blockchain.transactions`
     )
    SELECT COUNT(transaction_id) AS number,
       EXTRACT(month FROM trans_time) AS month,
       EXTRACT(day FROM trans_time) AS day
    FROM time
    WHERE EXTRACT(year FROM trans_time)=2018
    GROUP BY month,day
    ORDER BY month,day
""",max_gb_scanned=23)
transaction_by_day_2018
# _=plt.plot(transaction_by_day_2018.number)

plt.rcParams["figure.figsize"] = (16,8)
plt.plot(transaction_by_day_2018.number)
plt.title('Daily Number of Transcations in 2018')
import hashlib
import os

class DataSaver:
    def __init__(self, bitcoin_blockchain ):
        self.bitcoin_blockchain =bitcoin_blockchain 
        
    def Run_Query(self, query, max_gb_scanned=1):
        hashed_query=''.join(query.split()).encode("ascii","ignore")
        query_hash=hashlib.md5(hashed_query).hexdigest()
        query_hash+=".csv"
        if query_hash in os.listdir(os.getcwd()):
            print ("Data Already present getting it from file")
            return pd.read_csv(query_hash)
        else:
            data=self.bitcoin_blockchain .query_to_pandas_safe(query, max_gb_scanned=max_gb_scanned)
            data.to_csv(query_hash, index=False,encoding='utf-8')
            return data
        
bq=DataSaver(bitcoin_blockchain )
q = """
SELECT  o.output_pubkey_base58, sum(o.output_satoshis) as output_sum FROM
    `bigquery-public-data.bitcoin_blockchain.transactions` JOIN
    UNNEST(outputs) as o 
    where o.output_pubkey_base58 not in (select i.input_pubkey_base58
    from UNNEST(inputs) as i)
    group by o.output_pubkey_base58 order by output_sum desc limit 1000
"""

print (str(round((bitcoin_blockchain.estimate_query_size(q)),2))+str(" GB"))

results=bq.Run_Query(q, max_gb_scanned=70)
results["output_sum"]=results["output_sum"].apply(lambda x: float(x/100000000))
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
objects = results["output_pubkey_base58"][:20]
y_pos = np.arange(len(objects))
performance = results["output_sum"][:20]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)
plt.ylabel('Bitcoins')
plt.title('Bitcoins Addresses Who received Most number of bitcoins')
plt.show()
