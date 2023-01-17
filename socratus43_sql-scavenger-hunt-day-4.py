# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
#Lets build our query using WITH
query1 = """WITH temp AS
             (SELECT TIMESTAMP_MILLIS(timestamp) as trans_time,
                transaction_id
              FROM `bigquery-public-data.bitcoin_blockchain.transactions`
              )
            SELECT EXTRACT(DAY FROM trans_time) as day,
                EXTRACT(MONTH FROM trans_time) as month,
                EXTRACT(YEAR FROM trans_time) as year,
                COUNT(trans_time) as number
            FROM temp
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day, month, year
            ORDER BY month, day, year DESC
            """
per_day_2017 = bitcoin_blockchain.query_to_pandas(query1)
per_day_2017.head()


#May make image to see how it's look like
import matplotlib.pyplot as plt
plt.plot(per_day_2017.number, color = 'r')
plt.show()
#A little bit haotic
bitcoin_blockchain.head("transactions")
#Query
query2 = """SELECT COUNT(merkle_root) as number, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY number DESC
            """
merkle_roots = bitcoin_blockchain.query_to_pandas(query2)
merkle_roots.head()
#Some graph
mean = merkle_roots["number"].mean()
plt.plot(merkle_roots.number)
plt.hlines(mean, xmin = 0, xmax = 500000, color = 'g')
plt.ylim(0,13000)
plt.title("Comparing with average transactions per root\n(green line)")
plt.show()