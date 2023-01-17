import numpy as np
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
#For plots
import matplotlib.pyplot as plt
bc = BigQueryHelper('bigquery-public-data', 'bitcoin_blockchain')
bc_tables = bc.list_tables()
print("There are "+str(len(bc_tables))+" tables in the dataset")
print(bc_tables)
for x in range(0,len(bc_tables)):
    print("Table: "+bc_tables[x])
    a=bc.table_schema(bc_tables[x])
    for y in range(0,len(a)):
        print(a[y])
    print("\n\r")
sql1="""WITH time AS
            (SELECT TIMESTAMP_MILLIS(timestamp) AS ts, transaction_id as td
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            WHERE EXTRACT(year FROM TIMESTAMP_MILLIS(timestamp))=2017)
        SELECT EXTRACT(dayofyear FROM ts) AS doy, count(td) as ctd
        FROM time
        GROUP BY doy
        ORDER BY doy
    """
bc.estimate_query_size(sql1)
trans = bc.query_to_pandas(sql1)
trans.shape
#Looking at the first and last 5 entries (days of the year) we see
print(trans.head(5))
print(trans.tail(5))
#And plotting to see progress over the year 2017
plt.plot(trans.ctd)
plt.title("Number of BitCoin transactions over the year 2017")
sql2="""SELECT COUNT(transaction_id) AS cnt, merkle_root AS mr
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY mr
        ORDER BY cnt DESC
    """
bc.estimate_query_size(sql2)
merkle = bc.query_to_pandas(sql2)
merkle.shape
print(merkle.head(5))
print(merkle.tail(5))
isone=merkle.loc[merkle['cnt'] == 1]
print("Only sold once:")
print(isone.shape[0])
plt.plot(merkle.cnt)
plt.title("Individual counts for Merkle_Roots")
plt.show()
plt.plot(np.log(merkle.cnt))
plt.title("log(individual counts) for Merkle_Roots")
plt.show()