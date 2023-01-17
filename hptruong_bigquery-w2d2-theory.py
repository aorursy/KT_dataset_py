# Good day, Mr. Truong. Welcome back! How's your job?

# Lots of things going on. I still feel confused a lot about many things like the organization, etc.

# Ah, it's typical, it's common, and it's a double-sword.

# What do you mean?

# Well, kids nowadays get discouraged very easily. For example, lots of people you see always say

# I want to do this, I want to that, Can we do this, I want to do a kick-ass project, only fail to provide

# an actionable write-up--a concrete goal or proposal that you can stick to.

# Hmmm... this happens to me many times.

# Anyways, today we'll learn about AS for aliasing and WITH... AS for readability through the use of CTE

# To get started, load the dataset `crypto_bitcoin`. Hope you can still remember the code.

# Yes, sir!



from google.cloud import bigquery # talk to the sales guy

client = bigquery.Client() # create an account

dataset_ref = client.dataset('crypto_bitcoin', project='bigquery-public-data') # create a handler to the dataset

dataset = client.get_dataset(dataset_ref) # fetch the dataset through an API request

table_ref = dataset_ref.table('transactions') # create a handler to the `crypto_bitcoin` table

trans = client.get_table(table_ref) # fetch the table through an API request

client.list_rows(trans, max_results=5).to_dataframe()
# Feeling shaky?

# Kind of. But hey, I got the job done.

# Hmmm... I'll let you get away from it this time.

# Now query the number of transactions per day, sorted by date. Remember to use CTE for practice purpose

query_with_CTE = """

                 WITH time AS

                 (

                     SELECT DATE(block_timestamp) AS trans_date

                     FROM `bigquery-public-data.crypto_bitcoin.transactions`

                 )

                 SELECT trans_date

                       ,COUNT(trans_date) AS NumTrans

                 FROM time

                 GROUP BY trans_date

                 ORDER BY trans_date

                 """

# Set up query quota, cancel if exceeds 10GB

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)

safe_query_job = client.query(query_with_CTE, job_config=safe_config)

transactions_by_date = safe_query_job.to_dataframe()

transactions_by_date.head()
# Plot the nubmer of transactions

import matplotlib.pyplot as plt



transactions_by_date.set_index('trans_date').plot()

plt.show()