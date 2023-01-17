import bq_helper

bitcoin_blockchain = bq_helper.BigQueryHelper(
    active_project="bigquery-public-data",
    dataset_name="bitcoin_blockchain"
)
query_transactions2017_per_day = """
    WITH trns AS
    (
        SELECT TIMESTAMP_MILLIS(timestamp) AS trns_time
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
    SELECT  EXTRACT(MONTH from trns_time) AS month,
            EXTRACT(DAY from trns_time) AS day,
            COUNT(trns_time) as count
    FROM trns
    WHERE EXTRACT(YEAR from trns_time) = 2017
    GROUP BY month, day
    ORDER BY month, day
"""
transactions2017_per_day = bitcoin_blockchain.query_to_pandas_safe(
    query_transactions2017_per_day,
    max_gb_scanned=3
)
transactions2017_per_day
query_id_per_merkle_root = """
    SELECT merkle_root,
           COUNT(block_id) AS blocks
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle_root
"""
id_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(
    query_id_per_merkle_root,
    max_gb_scanned=37
)
id_per_merkle_root
#bitcoin_blockchain.head('blocks')