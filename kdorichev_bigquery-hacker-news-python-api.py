from google.cloud import bigquery

client = bigquery.Client()
PROJECT = "bigquery-public-data"

DATASET = "hacker_news"

dataset_ref = client.dataset(DATASET, project=PROJECT)

dataset = client.get_dataset(dataset_ref)
dataset.description
dataset.to_api_repr()
# List the tables in dataset

tables = list(client.list_tables(dataset))

for t in tables:

    print(t.table_id)
# Check the schema of one table

comments_table = client.get_table(PROJECT+'.'+DATASET+'.'+tables[0].table_id)

comments_table.schema
for f in comments_table.schema:

    print(f'{f.name}\t{f.field_type}\t{f.description}')
# Table field in a dictionary form

comments_table.schema[5].to_api_repr()
# Preview the first five lines of the "comments" table as Pandas DataFrame

df = client.list_rows(comments_table, max_results=5).to_dataframe()

df
# Limit the query results to 1 GB

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)
# Query to select prolific commenters and post counts

prolific_commenters_query = """

        SELECT author, COUNT(1) AS NumPosts

        FROM `bigquery-public-data.hacker_news.comments`

        GROUP BY author

        HAVING COUNT(1) > 10000

        ORDER BY NumPosts DESC

        """
query_job = client.query(prolific_commenters_query, job_config=safe_config)
if query_job.done(): 

    prolific_commenters = query_job.to_dataframe()

prolific_commenters.head()
deleted_posts_query = """

                      SELECT COUNT(1) AS num_deleted_posts

                      FROM `bigquery-public-data.hacker_news.comments`

                      WHERE deleted = True

                      """

query_job = client.query(deleted_posts_query)



# API request - run the query, and return a pandas DataFrame

if query_job.done: deleted_posts = query_job.to_dataframe()



# View results

print(deleted_posts)