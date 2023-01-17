from google.cloud import bigquery
import pandas as pd
from bq_helper import BigQueryHelper
import numpy as np 
import pandas as pd 
bitcoin = BigQueryHelper("bigquery-public-data","bitcoin_blockchain")

bitcoin.list_tables()
bitcoin.table_schema("blocks")
bitcoin.head("blocks")
bitcoin.table_schema("transactions")
bitcoin.head("transactions").transaction_id
bitcoin.head("transactions")

bitcoin.table_schema("transactions")
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

query = """
#standardSQL
  SELECT
    ARRAY_LENGTH(transactions) NoOfTrans,
    block_id 
  FROM
    `bigquery-public-data.bitcoin_blockchain.blocks`
ORDER BY
  NoOfTrans desc
LIMIT 10
"""
query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
result = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
result.columns
import seaborn as sns; 
import matplotlib.pyplot as plt
sns.set()
sns.barplot(x="NoOfTrans", y="block_id", data=result)
plt.show()


from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

query = """
#standardSQL
SELECT
  o.day,
  COUNT(o.transaction_id) AS NoOfTrans
FROM (
  SELECT
    TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day,
    transaction_id
  FROM
    `bigquery-public-data.bitcoin_blockchain.transactions` ) AS o
GROUP BY
  day
ORDER BY
NoOfTrans desc
"""
query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
result2 = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
result2.head()

sns.jointplot(x="day",y="NoOfTrans", data=result2,kind="kde", color="m");

type(result2['day'][0])

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

result2['datetime'] = pd.to_datetime(result2['day'],format="%d.%m.%Y %H:%M:%S.%f")
result2.set_index('datetime', inplace=True)

x = result2['NoOfTrans'].plot()

ticklabels = result2.index.strftime('%Y-%m-%d')
ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.show()

