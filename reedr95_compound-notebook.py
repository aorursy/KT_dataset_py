from google.cloud import bigquery
import matplotlib.pyplot as plt
import pandas as pd

pd.options.display.float_format = '{:.6f}'.format

client = bigquery.Client()

query = """
SELECT logs.block_number, logs.block_timestamp, topic,
  CONCAT('0x', SUBSTR(logs.data,1+2+24,40))     AS account,
  CONCAT('0x', SUBSTR(logs.data,1+2+64+24,40))  AS asset,
  SUBSTR(logs.data,1+2+128,64)                  AS amount,
  SUBSTR(logs.data,1+2+192,64)                  AS startingBalance,
  SUBSTR(logs.data,1+2+256,64)                  AS newBalance
FROM
  `bigquery-public-data.ethereum_blockchain.transactions` AS transactions,
  `bigquery-public-data.ethereum_blockchain.logs` AS logs JOIN UNNEST(topics) AS topic
WHERE TRUE
  AND transactions.hash = logs.transaction_hash
  AND address = to_address
  --AND topic = '0x4ea5606ff36959d6c1a24f693661d800a98dd80c0fb8469a665d2ec7e8313c21'
  AND to_address = '0x3fda67f7583380e67ef93072294a7fac882fd7e7'
ORDER BY block_number
"""

query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Event Topics:
# SupplyReceived: 0x4ea5606ff36959d6c1a24f693661d800a98dd80c0fb8469a665d2ec7e8313c21
# SupplyWithdrawn: 0x56559a17e3aa8ea4b05036eaf31aeaf9fb71fc1b8865b6389647639940bed030
# BorrowTaken: 0x6b69190ebbb96f162b04dc222ef96416f9dca9a415b6dd183c79424501113e18
# BorrowRepaid: 0x550e7e464126359c6adc43831f011682856b177df6c49c0af6675dd2a063649d
# Failure: 0x45b96fe442630264581b197e84bbada861235052c5a1aadfff9ea4e40a969aa0

# drop rows that aren't supplying WETH (https://weth.io/)
df = df[(df['asset'] == '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2')]

def hex_to_eth(x):
    return int(x, 16) / 1000000000000000000

# convert hex strings to integers and convert from wei to eth
df['amount'] = df['amount'].apply(hex_to_eth)
df['startingBalance'] = df['startingBalance'].apply(hex_to_eth)
df['newBalance'] = df['newBalance'].apply(hex_to_eth)

# drop rows that aren't logging the "SupplyReceived" event
supplies_received = df[(df['topic'] == '0x4ea5606ff36959d6c1a24f693661d800a98dd80c0fb8469a665d2ec7e8313c21')]
supplies_withdrawn = df[(df['topic'] == '0x56559a17e3aa8ea4b05036eaf31aeaf9fb71fc1b8865b6389647639940bed030')]
borrows_taken = df[(df['topic'] == '0x6b69190ebbb96f162b04dc222ef96416f9dca9a415b6dd183c79424501113e18')]
supplies_received.drop(columns=['asset', 'topic'])
supplies_withdrawn.drop(columns=['asset', 'topic'])
total_supply = supplies_received['amount'].sum() - supplies_withdrawn['amount'].sum()
total_supply
received_dates  = [pd.to_datetime(d) for d in supplies_received['block_timestamp']]
withdrawn_dates = [pd.to_datetime(d) for d in supplies_withdrawn['block_timestamp']]

plt.rcParams['figure.figsize'] = [15, 10]
plt.scatter(x=received_dates, y=supplies_received['amount'], s=supplies_received['amount'], label='WETH Supply Received')
plt.scatter(x=withdrawn_dates, y=supplies_withdrawn['amount'], s=supplies_withdrawn['amount'], c='red', label='WETH Supply Withdrawn')
legend = plt.legend()
for handle in legend.legendHandles:
    handle.set_sizes([10.0])
len(supplies_received)
len(borrows_taken)