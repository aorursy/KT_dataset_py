# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper
# bigQueryデータセットのヘルパーオブジェクト作成
blockchain_helper = BigQueryHelper(active_project="bigquery-public-data",dataset_name="bitcoin_blockchain")
# テーブル一覧
blockchain_helper.list_tables()
blockchain_helper.head('blocks')
blockchain_helper.head('transactions')
# 年月日に変換
# 日別にブロック数を取得
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    block_id
                FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            )
            SELECT COUNT(block_id) AS blocks,
                EXTRACT(DATE FROM trans_time) AS date
            FROM time
            GROUP BY date
            ORDER BY date
        """
# クエリ実行の際のサイズ（GB）を出力 
blockchain_helper.estimate_query_size(query)
# query_to_pandas_safe()：サイズが1GB以上のクエリは実行されない
q1 = blockchain_helper.query_to_pandas_safe(query) 
q1.head(5)

q1.tail(5)
plt.bar(q1['date'], q1['blocks'])
plt.show()
# ascending=False：降順
q1.sort_values('blocks', ascending=False).head(10)
query = """WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month
            ORDER BY year, month
            """
# check size of query
blockchain_helper.estimate_query_size(query)

q4 = blockchain_helper.query_to_pandas(query)
import matplotlib.pyplot as plt
plt.plot(q4.transactions)
plt.title("Monthly Bitcoin Transactions")
plt.show()
q4.sort_values('transactions', ascending=False).reset_index(drop=True).head()
query = """WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year
            ORDER BY year
            """
# check size of query
blockchain_helper.estimate_query_size(query)
q5 = blockchain_helper.query_to_pandas(query) 
import matplotlib.pyplot as plt
plt.plot(q5.transactions)
plt.title("Every Year Bitcoin Transactions")
plt.show()
q5.sort_values('transactions', ascending=False).reset_index(drop=True)
query = """
SELECT  o.output_pubkey_base58, sum(o.output_satoshis) as output_sum from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o 
    where o.output_pubkey_base58 not in (select i.input_pubkey_base58
    from UNNEST(inputs) as i)
    group by o.output_pubkey_base58 order by output_sum desc limit 10
"""
print (str(round((blockchain_helper.estimate_query_size(query)),2))+str(" GB"))
q6 = blockchain_helper.query_to_pandas(query) 
#1bitcoin=100,000,000satoshi
q6["output_sum"]=q6["output_sum"].apply(lambda x: float(x/100000000))
q6
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
objects = q6["output_pubkey_base58"]
y_pos = np.arange(len(objects))
performance = q6["output_sum"][:10]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)
plt.ylabel('Bitcoins')
plt.title('Bitcoins Addresses Who received Most number of bitcoins')
plt.show()
query = """
        WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    inputs.input_pubkey_base58 AS input_key,
                    outputs.output_pubkey_base58 AS output_key,
                    outputs.output_satoshis AS satoshis,
                    transaction_id AS trans_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    JOIN UNNEST (inputs) AS inputs
                    JOIN UNNEST (outputs) AS outputs
                WHERE outputs.output_pubkey_base58 = '1LNWw6yCxkUmkhArb2Nf2MPw6vG7u5WG7q'
            )
        SELECT input_key, output_key, satoshis, trans_id,
            EXTRACT(DATE FROM trans_time) AS date
        FROM time
        --ORDER BY date
        """

print (str(round((blockchain_helper.estimate_query_size(query)),2))+str(" GB"))
q7 = blockchain_helper.query_to_pandas(query) 
# make a datatime type transformation
q7['date'] = pd.to_datetime(q7.date)
q7 = q7.sort_values('satoshis',ascending=False)
# convert satoshis to bitcoin
q7['bitcoin'] = q7['satoshis'].apply(lambda x: float(x/100000000))
q7
q7.shape
q7_mod=q7.loc[:,"date":"bitcoin"]
q7_mod.groupby([q7_mod['date'].dt.year, q7_mod['date'].dt.month]).sum()

QUERY = """
SELECT
    inputs.input_pubkey_base58 AS input_key, count(*)
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    JOIN UNNEST (inputs) AS inputs
WHERE inputs.input_pubkey_base58 IS NOT NULL
GROUP BY inputs.input_pubkey_base58 order by count(*) desc limit 1000
"""
blockchain_helper.estimate_query_size(QUERY) 
q8 = blockchain_helper.query_to_pandas(QUERY) 
q8.head()
# lets query all transactions this person was involved in
q_input = """
        WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    inputs.input_pubkey_base58 AS input_key,
                    outputs.output_pubkey_base58 AS output_key,
                    outputs.output_satoshis AS satoshis,
                    transaction_id AS trans_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    JOIN UNNEST (inputs) AS inputs
                    JOIN UNNEST (outputs) AS outputs
                WHERE inputs.input_pubkey_base58 = '1NxaBCFQwejSZbQfWcYNwgqML5wWoE3rK4'
            )
        SELECT input_key, output_key, satoshis, trans_id,
            EXTRACT(DATE FROM trans_time) AS date
        FROM time
        --ORDER BY date
        """
blockchain_helper.estimate_query_size(q_input)
xxx = blockchain_helper.query_to_pandas(q_input)

q9=xxx
q9.head()
# タイムスタンプを年月日に変換
q9['date'] = pd.to_datetime(q9.date)
q9 = q9.sort_values('date')
q9['bitcoin'] = q9['satoshis'].apply(lambda x: float(x/100000000))
q9.head()
q9.tail()
q9[["bitcoin"]].describe()
len(q9)
# input_key と output_key が同じ
q9_same=q9[q9["input_key"]==q9["output_key"]]
len(q9_same)
# 1日あたりの取引回数
q9['date'].value_counts().head()
plt.bar(q9.index, q9.values)
plt.show()


