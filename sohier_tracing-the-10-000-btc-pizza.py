import pandas as pd

from bq_helper import BigQueryHelper
from sys import getsizeof
bq_assist = BigQueryHelper('bigquery-public-data', 'bitcoin_blockchain')
bq_assist.head('transactions', num_rows=3)
QUERY_TEMPLATE = """
SELECT
    timestamp,
    inputs.input_pubkey_base58 AS input_key,
    outputs.output_pubkey_base58 AS output_key,
    outputs.output_satoshis as satoshis
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    JOIN UNNEST (inputs) AS inputs
    JOIN UNNEST (outputs) AS outputs
WHERE inputs.input_pubkey_base58 IN UNNEST({0})
    AND outputs.output_satoshis  >= {1}
    AND inputs.input_pubkey_base58 IS NOT NULL
    AND outputs.output_pubkey_base58 IS NOT NULL
GROUP BY timestamp, input_key, output_key, satoshis
"""
def trace_transactions(target_depth, seeds, min_satoshi_per_transaction, bq_assist):
    """
    Trace transactions associated with a given bitcoin key.

    To limit the number of BigQuery calls, this function ignores time. 
    If you care about the order of transactions, you'll need to do post-processing.

    May return a deeper graph than the `target_depth` if there are repeated transactions
    from wallet a to b or or self transactions (a -> a).
    """
    MAX_SEEDS_PER_QUERY = 500
    query = QUERY_TEMPLATE.format(seeds, min_satoshi_per_transaction)
    print(f'Estimated total query size: {int(bq_assist.estimate_query_size(query)) * MAX_DEPTH}')
    results = []
    seeds_scanned = set()
    for i in range(target_depth):
        seeds = seeds[:MAX_SEEDS_PER_QUERY]
        print(f"Now scanning {len(seeds)} seeds")
        query = QUERY_TEMPLATE.format(seeds, min_satoshi_per_transaction)
        transactions = bq_assist.query_to_pandas(query)
        results.append(transactions)
        # limit query kb by dropping any duplicated seeds
        seeds_scanned.update(seeds)
        seeds = list(set(transactions.output_key.unique()).difference(seeds_scanned))
    return pd.concat(results).drop_duplicates()
MAX_DEPTH = 4
BASE_SEEDS = ['1XPTgDRhN8RFnzniWCddobD9iKZatrvH4']
#SATOSHI_PER_BTC = 10**7
#MIN_BITCOIN_PER_TRANSACTION = 1/1000
#min_satoshi = MIN_BITCOIN_PER_TRANSACTION * SATOSHI_PER_BTC
df = trace_transactions(MAX_DEPTH, BASE_SEEDS, 0, bq_assist)
df.to_csv("transactions.csv", index=False)
df.head()
df.info()
