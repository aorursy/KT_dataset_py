from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "crypto_ethereum" dataset (https://www.kaggle.com/bigquery/ethereum-blockchain)

dataset_ref = client.dataset("crypto_ethereum", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# List all the tables in the "crypto_ethereum" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there's only one!)

for table in tables:  

    print(table.table_id)
# This table show us the most extreme cases for regular contract calls in ETH 



top10_receipt_gas_used_query = """

        SELECT `hash`, `value`, `gas_price` as `gas_price_in_wei`, `receipt_gas_used`, `receipt_cumulative_gas_used`, `gas_price` / 1000000000000000000 * `receipt_gas_used` as fee_in_ether, `block_timestamp`

        FROM `bigquery-public-data.crypto_ethereum.transactions`

        ORDER BY `receipt_gas_used` DESC

        LIMIT 10

        """

top10_receipt_gas_used_job = client.query(top10_receipt_gas_used_query)

top10_receipt_gas_used_response = top10_receipt_gas_used_job.to_dataframe()



top10_receipt_gas_used_response

# This table show us the most extreme cases for regular transfers in ETH



top10_gas_price_query = """

        SELECT `hash`,`value`, `gas_price` as `gas_price_in_wei`, `receipt_gas_used`, `receipt_cumulative_gas_used`, `gas_price` / 1000000000000000000 * `receipt_gas_used` as fee_in_ether, `block_timestamp`

        FROM `bigquery-public-data.crypto_ethereum.transactions`

        ORDER BY `gas_price` DESC

        LIMIT 10

        """

top10_gas_price_job = client.query(top10_gas_price_query)

top10_gas_price_response = top10_gas_price_job.to_dataframe()

top10_gas_price_response
# Tx hash of the top one

# https://etherscan.io/tx/0x1f73b43dc9c48cc131a931fac7095de9e5eba0c5184ec0c5c5f1f32efa2a6bab

print(top10_gas_price_response.hash.head()[3])
# This table show us the most extreme cases for contract calls that also transfer value in ETH



# notice that "up_front_cost = gas_price * gas_limit + value" and here we are using gas_used instead of gas_limit

top10_value_transfer_in_contract_query = """

        SELECT `hash`, `value`, `gas_price` as `gas_price_in_wei`, `receipt_gas_used`, `receipt_cumulative_gas_used`, `gas_price` / 1000000000000000000 * `receipt_gas_used` as fee_in_ether, (`gas_price` / 1000000000000000000 * `receipt_gas_used`) + `value` / 1000000000000000000 as up_front_cost_aprox_in_ether, `block_timestamp`

        FROM `bigquery-public-data.crypto_ethereum.transactions`

        WHERE `receipt_gas_used` > 21000

        ORDER BY `value` DESC

        LIMIT 10

        """

top10_value_transfer_in_contract_job = client.query(top10_value_transfer_in_contract_query)

top10_value_transfer_in_contract_response = top10_value_transfer_in_contract_job.to_dataframe()



top10_value_transfer_in_contract_response