import pandas as pd
import bq_helper
import matplotlib.pyplot as plt
bitcoin_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                       dataset_name="bitcoin_blockchain")

# Listing tables
print(bitcoin_data.list_tables())

# Showing their schema
print("Schema for "+bitcoin_data.list_tables()[0])
print(bitcoin_data.table_schema(bitcoin_data.list_tables()[0]))
print("\n\n")
print("Schema for "+bitcoin_data.list_tables()[1])
print(bitcoin_data.table_schema(bitcoin_data.list_tables()[1]))

# I will first create a useful subset using WITH CTE and then run the group by command on that
query = """
WITH tr_useful AS (
SELECT
    TIMESTAMP_MILLIS(timestamp) AS Time, -- To make the integer as timestamp so we can extract year later
    transaction_id
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT
    EXTRACT(DAY FROM Time) AS Day_Num,
    COUNT(transaction_id) AS Num_Transactions
FROM tr_useful
WHERE EXTRACT(YEAR FROM Time) = 2017
GROUP BY Day_Num
ORDER BY Day_Num
"""

trans_2017 = bitcoin_data.query_to_pandas_safe(query,max_gb_scanned=21)

# This data is too good not to plot to see trend over time. Since it is sorted by day number, we can 
# directly plot
print(trans_2017.head(n=5))
plt.plot(trans_2017['Num_Transactions'])
plt.title('Bitcoin Transactions in 2017')
# Since this query seems too simple, to be using a WITH, I'm going to restrict even this to 2017 and 
# my CTE will be a dataset restricted to 2017
query2 = """
WITH tr_2017 AS (
SELECT
    merkle_root,
    transaction_id
FROM `bigquery-public-data.bitcoin_blockchain.transactions`    
WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp))=2017
)
SELECT
    merkle_root,
    COUNT(transaction_id) AS Num_Transactions
FROM tr_2017
GROUP BY merkle_root
ORDER BY Num_Transactions DESC
"""

merkle_tr = bitcoin_data.query_to_pandas_safe(query2,max_gb_scanned=40)
merkle_tr.head(n=10)