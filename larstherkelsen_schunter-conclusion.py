import numpy as np
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
#For plots
import matplotlib.pyplot as plt
bc = BigQueryHelper('bigquery-public-data', 'bitcoin_blockchain')
bc_tables = bc.list_tables()
sql1="""SELECT TIMESTAMP_MILLIS(timestamp) AS ts, merkle_root AS mr,
        EXTRACT(year FROM TIMESTAMP_MILLIS(timestamp)) AS year,
        EXTRACT(month FROM TIMESTAMP_MILLIS(timestamp)) AS month,
        EXTRACT(week FROM TIMESTAMP_MILLIS(timestamp)) AS week,
        EXTRACT(dayofyear FROM TIMESTAMP_MILLIS(timestamp)) AS dayofyear,
        EXTRACT(day FROM TIMESTAMP_MILLIS(timestamp)) AS day
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        WHERE merkle_root IN
            (WITH cntmro AS
                (SELECT merkle_root, COUNT(transaction_id) as ts
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root)
            SELECT DISTINCT merkle_root
            FROM cntmro
            WHERE ts=1)
        ORDER BY ts DESC"""
bc.estimate_query_size(sql1)

onetrans = bc.query_to_pandas(sql1)
onetrans.to_csv("onetrans.csv")
onetrans.shape
onetrans['cnt']=1
onetrans
abc=onetrans.groupby(['year','month'])['cnt'].value_counts()

onetrans['cnt']=1
onetrans
abc=pd.DataFrame(onetrans.groupby(['year','month'])['cnt'].value_counts().unstack())
cde=pd.DataFrame(columns=['Time','Count'])
for i in range(2009,2018):
    for j in range(1,12):
        a=str(i)+"_"+str(j)
        b=abc[1][i][j]
        c=cde['Time'].count()
        cde.loc[c]=[a,b]
cde
cde.plot.line()
plt.show()
sql2="""SELECT output.output_pubkey_base58 AS output_key,
        COUNT(merkle_root) AS mr_cnt
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`,
        UNNEST(outputs) AS output
        WHERE merkle_root IN
            (WITH cntmro AS
                (SELECT merkle_root, COUNT(transaction_id) as ts
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root)
            SELECT DISTINCT merkle_root
            FROM cntmro
            WHERE ts=1)
        GROUP BY output_key
        ORDER BY mr_cnt DESC"""
bc.estimate_query_size(sql2)

keycount = bc.query_to_pandas(sql2)
keycount.to_csv("keycount.csv")
keycount.shape
print(keycount.head(5))
print("There are "+str(keycount["mr_cnt"].sum())+" instances")
print("of those "+str(keycount["mr_cnt"][0])+" have no output adress")
print("The top 100 of the rest have hoarded Bitcoins as seen below")
keycount1=keycount.drop(keycount.index[[0]])
keycount1.head(30).plot.line()
plt.show()