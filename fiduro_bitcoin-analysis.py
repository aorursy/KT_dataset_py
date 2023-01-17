#Class to fetch Data if not present and save it. A following retrieval won't be deducted from quota
import pandas as pd
import hashlib
import os

class DataSaver:
    def __init__(self, bq_assistant):
        self.bq_assistant=bq_assistant
        
    def Run_Query(self, query, max_gb_scanned=1):
        hashed_query=''.join(query.split()).encode("ascii","ignore")
        query_hash=hashlib.md5(hashed_query).hexdigest()
        query_hash+=".csv"
        if query_hash in os.listdir(os.getcwd()):
            print ("Data Already present getting it from file")
            print (query_hash)
            return pd.read_csv(query_hash)
        else:
            data=self.bq_assistant.query_to_pandas_safe(query, max_gb_scanned=max_gb_scanned)
            data.to_csv(query_hash, index=False,encoding='utf-8')
            print (query_hash)
            return data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)from google.cloud import bigquery

from google.cloud import bigquery
from bq_helper import BigQueryHelper
client = bigquery.Client()
bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")
def satoshi_to_bitcoin(satoshi):
    return float(float(satoshi)/ float(100000000))
bq=DataSaver(bq_assistant)

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

test_data=bq_assistant.head('transactions', num_rows=100)
#TEST
q = """
SELECT TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day, count(transaction_id) as txs_count from 
    `bigquery-public-data.bitcoin_blockchain.transactions` GROUP BY day ORDER BY day
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

#COUNT of transactions
q = """
SELECT TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day, count(transaction_id) as txs_count from 
    `bigquery-public-data.bitcoin_blockchain.transactions` GROUP BY day ORDER BY day
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

results2=bq.Run_Query(q, max_gb_scanned=25)
results2.head()
#COUNT of transactions
trace1 = go.Scatter(
                    x = results2.day,
                    y = results2.txs_count,
                    mode = "lines",
                    name = "transactions")
data = [trace1]
layout = dict(title = 'BTC Transactions per day - Cleaned',
              yaxis= dict(title= '# Transactions',zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
#AVERAGE transacted BTC per day
#f8219d6d11cea8a8e0b04452352bdf9f.csv
q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day, avg(o.output_satoshis) as output_avg from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o group by day order by day
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

results3=bq.Run_Query(q, max_gb_scanned=10)

#CONVERT SATOSHIS TO BITCOINS
results3["output_avg"]=results3["output_avg"].apply(lambda x: float(x/100000000))
results3.head()
#AVERAGE transacted BTC per day
trace1 = go.Scatter(
                    x = results3.day,
                    y = results3.output_avg,
                    mode = "lines",
                    name = "average value")
data = [trace1]
layout = dict(title = 'BTC average transaction value',
              yaxis= dict(title= '# Transactions',zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
#AVERAGE sending addresses per transaction
q = """
    SELECT TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day, avg(s_addr_count) AS avg_s_addr_count FROM (
        SELECT max(timestamp) AS timestamp, transaction_id AS transaction_id, count(i.input_pubkey_base58) AS s_addr_count FROM `bigquery-public-data.bitcoin_blockchain.transactions` JOIN UNNEST(inputs) as i GROUP BY transaction_id
    ) GROUP BY day ORDER by day
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

results4=bq.Run_Query(q, max_gb_scanned=50)
results4.head()
#AVERAGE sending addresses per transaction
trace1 = go.Scatter(
                    x = results4.day,
                    y = results4.avg_s_addr_count,
                    mode = "lines",
                    name = "average value")
data = [trace1]
layout = dict(title = 'Average sending addresses per BTC transaction',
              yaxis= dict(title= '# Transactions',zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)