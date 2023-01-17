from google.cloud import bigquery



client = bigquery.Client()

# If you query a BQ dataset that wasn't noboard, link you GCP account and setup the bigquery.Client() as follows:

# PROJECT_TO_RUN_JOBS = 'my-project-to-run-queries'

# client = bigquery.Client(project=PROJECT_TO_RUN_JOBS)



# List the tables in geo_openstreetmap dataset which resides in bigquery-public-data project:

dataset = client.get_dataset('bigquery-public-data.geo_openstreetmap')

tables = list(client.list_tables(dataset))

print([table.table_id for table in tables])
sql = '''

SELECT nodes.*

FROM `bigquery-public-data.geo_openstreetmap.planet_nodes` AS nodes

JOIN UNNEST(all_tags) AS tags

WHERE tags.key = 'amenity'

  AND tags.value IN ('hospital',

    'clinic',

    'doctors')

LIMIT 10

'''

# Set up the query

query_job = client.query(sql)



# Make an API request  to run the query and return a pandas DataFrame

df = query_job.to_dataframe()

df.head(5)