# import the bigquery
from google.cloud import bigquery
# initializing the client object
client = bigquery.Client()
# asking client to take the reference
dataset_ref = client.dataset('hacker_news', project = 'bigquery-public-data')

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))

for i in tables:
    print(i.table_id)
# wee need only comments table

table_ref = dataset_ref.table('comments')
table = client.get_table(table_ref)
table.schema
client.list_rows(table, max_results = 5).to_dataframe()
query = """ select parent, count(id) from `bigquery-public-data.hacker_news.comments` 
            group by parent
            having count(id) > 10   
         """
# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)

# sending query to the database

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query, job_config = safe_config)


popular_comments = query_job.to_dataframe()
popular_comments.sort_values(by = ['f0_'], ascending = False).head()
query_improved = """
                 SELECT parent, COUNT(1) AS NumPosts
                 FROM `bigquery-public-data.hacker_news.comments`
                 GROUP BY parent
                 HAVING COUNT(1) > 10
                 """

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_improved, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
improved_df = query_job.to_dataframe()

# Print the first five rows of the DataFrame
improved_df.head()
improved_df.sort_values(by = ['NumPosts'], ascending = False).head(10)
query_good = """
             SELECT parent, COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY parent
             """
query_bad = """
            SELECT author, parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            """
