from google.cloud import bigquery

client = bigquery.Client()



Dataset_ref = client.dataset('hacker_news', project = 'bigquery-public-data')



dataset = client.get_dataset(Dataset_ref)
table_ref = Dataset_ref.table("comments")

# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "comments" table

client.list_rows(table, max_results=5).to_dataframe()
query = """

        SELECT parent,count(id) as num_of_posts

        from `bigquery-public-data.hacker_news.comments`

        GROUP BY parent

        HAVING count(id)>10

        """

#Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)



safe_config = bigquery.QueryJobConfig(maximum_byte_billed = 10**10)



query_job = client.query(query, job_config = safe_config)

query_result = query_job.to_dataframe()



query_result.head(10)
query = """

        SELECT author, parent,count(id) as num_of_posts

        from `bigquery-public-data.hacker_news.comments`

        GROUP BY parent,author

        HAVING count(id)>1

        """

#Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)



safe_config = bigquery.QueryJobConfig(maximum_byte_billed = 10**10)



query_job = client.query(query, job_config = safe_config)

query_result = query_job.to_dataframe()



query_result.head(10)
query = """

        SELECT count(1) as nun_of_deleted_comments

        from `bigquery-public-data.hacker_news.comments`

        WHERE deleted = True

        """

#Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)



safe_config = bigquery.QueryJobConfig(maximum_byte_billed = 10**10)



query_job = client.query(query, job_config = safe_config)

query_result = query_job.to_dataframe()



query_result