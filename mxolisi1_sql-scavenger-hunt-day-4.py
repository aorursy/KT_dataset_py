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
#*           checkin how BIG?
bitcoin_blockchain.estimate_query_size(query) #Yoh, 20.64Gigs

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)
#** How many Bitcoin transactions were made each day in 2017?
#** You can use the "timestamp" column from the "transactions" 
#**table to answer this question. You can check the notebook from Day 3 for more
#**information on timestamps.
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
                ,EXTRACT(Day FROM trans_time) AS day
            FROM time
            GROUP BY year, month ,day
            ORDER BY year, month , day
            
           -- HAVING (EXTRACT(YEAR FROM trans_time))='2017'
        """
#*           checkin how BIG?
bitcoin_blockchain.estimate_query_size(query1) #Yoh, 20.637550193816423~ 20.64Gigs

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
#              #* you may view the first 7 rows
transactions_per_day.head(7)

#*make 2017 ddata frame for the plot
#**transactions_in_2017= """select * from transactions_per_day where year= '2017'"""
transactions_in_2017= transactions_per_day[transactions_per_day.year == 2017]
#              #* you may view the first 7 rows, aswell
transactions_in_2017.head(7)

# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_in_2017.transactions)
plt.title("Daily Bitcoin Transcations in '2017'")
#*How many transactions are associated with each merkle root?
#*   oYou can use the "merkle_root" and "transaction_id" columns in the "transactions" table to 
#*answer this question.
#*   o Note that the earlier version of this question asked "How many blocks are associated with each
#*merkle root?", which would be one block for each root. Apologies for the confusion!
query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT DISTINCT merkle_root, COUNT(transaction_id) AS transactions
                --,EXTRACT(MONTH FROM trans_time) AS month,
                --EXTRACT(YEAR FROM trans_time) AS year
                --,EXTRACT(Day FROM trans_time) AS day
                 
            FROM time
            GROUP BY merkle_root --year, month ,day
             ORDER BY merkle_root--year, month , day
             --ORDER BY COUNT(transaction_id) DESC 
        """

#*           checkin how BIG?
bitcoin_blockchain.estimate_query_size(query2) #Yoh, 20.637550193816423~ 20.64Gigs/ 36.81292737275362~ 36.813Gigs

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)
#sort the df
# sorted_merkle_trans = df.sort(transactions_per_merkle.transactions, ascending=F)

#              #* you may view the first 7 rows
transactions_per_merkle.head(7)
#* sorting the df ()
test = transactions_per_merkle.sort(['transactions'], ascending=[False])
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_merkle.transactions)
plt.title("Bitcoin Transcations per merkle")