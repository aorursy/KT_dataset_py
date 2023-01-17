# Big Query

from google.cloud import bigquery



# Pandas

import numpy as np

import pandas as pd

#import pandas_gbq



# Others

# Create a "Client" object

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Dataset reference

print(type(dataset_ref))



# Dataset type

print(type(dataset))
# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)
# Construct a reference to the "full" table

table_ref = dataset.table("full")



# API request - fetch the table

bg_full = client.get_table(table_ref)
type(bg_full)
# Print information on all the columns in the "full" table in the "hacker_news" dataset

bg_full.schema
[command for command in dir(bg_full) if not command.startswith('_')]
schema_subset = [col for col in bg_full.schema if col.name in ('by', 'title', 'time')]

schema_subset
results = [x for x in client.list_rows(bg_full, start_index=100, selected_fields=schema_subset, max_results=10)]

print(results)
for i in results:

    print(dict(i))
# Table size in GB

BYTES_PER_GB = 2**30

bg_full.num_bytes / BYTES_PER_GB
# Function to estimate the query size in GB

def estimate_gigabytes_scanned(query, bq_client):

    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun

    my_job_config = bigquery.job.QueryJobConfig()

    my_job_config.dry_run = True

    my_job = bq_client.query(query, job_config=my_job_config)

    BYTES_PER_GB = 2**30

    return(my_job.total_bytes_processed / BYTES_PER_GB)
estimate_gigabytes_scanned("SELECT Id FROM `bigquery-public-data.hacker_news.stories`", client)
query1 = """SELECT parent, COUNT(id) FROM `bigquery-public-data.hacker_news.comments`

        limit = 1000

        """
query1 = """

                SELECT parent, COUNT(id) AS NumPosts

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

                """
estimate_gigabytes_scanned(query1, client)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query1, job_config=safe_config)
# API request - run the query, and convert the results to a pandas DataFrame

popular_comments = query_job.to_dataframe()
# Print the first 3 rows of the DataFrame

popular_comments.head(3)
type(popular_comments)
query2 = """

                SELECT parent, COUNT(id) AS NumPosts

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

                ORDER BY(parent)

                """
estimate_gigabytes_scanned(query2, client)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query2, job_config=safe_config)
# API request - run the query, and convert the results to a pandas DataFrame

popular_comments_order = query_job.to_dataframe()
# Print the first 3 rows of the DataFrame

popular_comments_order.head(3)
query3 = """

                WITH comment_order AS

                (SELECT parent, COUNT(id) AS NumPosts

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

                ORDER BY(parent))

                SELECT Max(parent) FROM comment_order

                """
estimate_gigabytes_scanned(query3, client)
query4 = """

        SELECT L.license, COUNT(1) AS number_of_files

        FROM `bigquery-public-data.github_repos.sample_files` AS sf

        INNER JOIN `bigquery-public-data.github_repos.licenses` AS L 

            ON sf.repo_name = L.repo_name

        GROUP BY L.license

        ORDER BY number_of_files DESC

        """
estimate_gigabytes_scanned(query4, client)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query4, job_config=safe_config)
# API request - run the query, and convert the results to a pandas DataFrame

file_count_by_license = query_job.to_dataframe()
file_count_by_license.head(3)
# Using WHERE reduces the amount of data scanned / quota used

query = """

SELECT *

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(k|K)aggle")

ORDER BY time

"""



query_job = client.query(query)



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

headlines = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the first 10 headlines

headlines.head(10)
import wordcloud

import matplotlib.pyplot as plt



words = ' '.join(headlines.title).lower()

cloud = wordcloud.WordCloud(background_color='black',

                            max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=300,

                            relative_scaling=.5).generate(words)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('kaggle-hackernews.png')

plt.imshow(cloud)

plt.show()