# Good day Mr. Truong. Welcome back! I'm very glad to see you again,

# because you show me your great consistency in learning about our service.

# I'm sorry, what's your service, again?

# ... It's bigquery, Mr. Truong.

# I was just kidding...

# Alright, as a punishment, show us what you remember!

# Okay.



# Call bigquery

from google.cloud import bigquery

# Sign up for the service

client = bigquery.Client()

# Access the hacker_news dataset in bigquery-public-data project

dataset_ref = client.dataset('hacker_news', project='bigquery-public-data') # handle

dataset     = client.get_dataset(dataset_ref) # fetch data

# Show the tables in dataset

tables      = client.list_tables(dataset)

for table in tables:

    print(table.table_id)

# Access table stories

table_ref   = dataset_ref.table("stories")

stories     = client.get_table(table_ref)

print(stories.schema)

# List 5 rows

client.list_rows(stories, max_results=5).to_dataframe()
# Good work. I understand that you were not kidding.

# Now, today we have two learning objectives

# (1) First, we'll show you how to submit a query job to our service (using your username, of course)

# (2) We have a limit to how many jobs you can submit (via memory limit). 

# Thus, we'll show you how to calculate the amount of memory that your query will take. 

# Even better, you can limit your query results by specifying the desired amout of processing memory



# Let's tackle objective (1) with the trial dataset, openaq.

# We're going to write a query logic that select all the cities in the U.S.

# First, what table(s) do we have in the openaq dataset?

from google.cloud import bigquery

client = bigquery.Client()

dataset_ref = client.dataset('openaq', project='bigquery-public-data')

dataset = client.get_dataset(dataset_ref)

tables = client.list_tables(dataset_ref)

for table in tables:

    print(table.table_id)

table_ref = dataset_ref.table('global_air_quality')

global_air_quality = client.get_table(table_ref)

print(global_air_quality.schema)

client.list_rows(global_air_quality, max_results=5).to_dataframe()
# Now, write a query to get all the cities in the U.S.

query = """

        SELECT city

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country='US'

        """

# Next, let's submit this query through your username.

# We use the default parameters here. Later, we'll change them later for advanced settings

query_job = client.query(query)

us_cities = query_job.to_dataframe()

us_cities.head()
# What five cities have the most measurement?

us_cities.city.value_counts().head()
# Well done! Moving on, let's calculate how much memory does this job take.

# Now, you won't have access to this service with your username.

# Thus, the first thing you have to do is signing up for this service, QueryJobConfig

query = """

        SELECT score, title

        FROM `bigquery-public-data.hacker_news.full`

        WHERE type='job'

        """

dry_run_config = bigquery.QueryJobConfig(dry_run=True)
# Now, you can have access to this service using your username

dry_run_query_job = client.query(query, job_config=dry_run_config)

print(dry_run_query_job)
# What returned to you is an object.

# We can use the attribute total_bytes_processed to calculate how much memory is needed

print('This query job will take {} bytes.'.format(dry_run_query_job.total_bytes_processed))
# Early stop

ONE_HUNDRED_MB = 100*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_HUNDRED_MB)

safe_query_job = client.query(query, job_config=safe_config)

safe_query_job.to_dataframe()