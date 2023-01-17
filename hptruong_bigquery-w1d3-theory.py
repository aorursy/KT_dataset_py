# Good morning, Mr. Truong.

# Today is the last day of our first summer camp event.

# I hope you've learned quite a lot about us. What's our name again?

# I don't... Just kidding (saw a frown from bigquery, telling me that I'd better me serious before having a punch in my gut). BIGQUERYYYY!

# Thanks for doing that. Unlike Bob, we're very serious about doing something. Remember, the fun only comes in after the work, not the other way around.

# Ok, ok. What's your point?

# Well, today we are learning more about GROUP BY and HAVING clauses.

# You already know what they do--good job on that, so I'd like you to figure which Hacker News comments generated the most discussion?

# Ok (I don't know who's the boss of who now...)





from google.cloud import bigquery

client = bigquery.Client()

dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')

dataset = client.get_dataset(dataset_ref)

tables = client.list_tables(dataset)

for table in tables:

    print(table.table_id)



table_ref = dataset_ref.table('comments')

table = client.get_table(table_ref)

print(table.schema)



client.list_rows(table, max_results=5).to_dataframe()
query = """

        SELECT parent, COUNT(id)

        FROM `bigquery-public-data.hacker_news.comments`

        GROUP BY parent

        HAVING COUNT(id) > 10

        """



# Set up the query (cancel the query if it would use too much of my quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query=query, job_config=safe_config)



popular_comments = query_job.to_dataframe()



popular_comments.head()
# Good job, Mr. Truong.

# With this, we can conclude our summer camp event, session 1. I look forward to seeing you in the next session.

# And don't forget to do the exercises and practice data wrangling.