from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "crypto_bitcoin" dataset

dataset_ref = client.dataset("crypto_bitcoin", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "transactions" table

table_ref = dataset_ref.table("transactions")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "transactions" table

client.list_rows(table, max_results=5).to_dataframe()
# Your Code Here

query = """ WITH time AS 

            (

                SELECT DATE(block_timestamp) AS trans_date

                FROM `bigquery-public-data.crypto_bitcoin.transactions`

                WHERE EXTRACT(YEAR FROM block_timestamp) = 2017

            )

            SELECT COUNT(1) AS transactions,

                trans_date

            FROM time

            GROUP BY trans_date

            ORDER BY trans_date

        """

# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

transactions_by_date = query_job.to_dataframe()



# Print the first five rows

transactions_by_date.head()
# Your Code Here

query = """ SELECT COUNT(block_number) AS transactions, 

                block_number AS block                

            FROM `bigquery-public-data.crypto_bitcoin.transactions`

            GROUP BY block            

"""

# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

transactions_by_block = query_job.to_dataframe()



# Print the first five rows

transactions_by_block.head(5)