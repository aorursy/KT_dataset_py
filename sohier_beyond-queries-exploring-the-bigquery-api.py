from google.cloud import bigquery
client = bigquery.Client()
hn_dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')
type(hn_dataset_ref)
hn_dset = client.get_dataset(hn_dataset_ref)
type(hn_dset)
[x.table_id for x in client.list_tables(hn_dset)]
hn_full = client.get_table(hn_dset.table('full'))
type(hn_full)
[command for command in dir(hn_full) if not command.startswith('_')]
hn_full.schema
schema_subset = [col for col in hn_full.schema if col.name in ('by', 'title', 'time')]
results = [x for x in client.list_rows(hn_full, start_index=100, selected_fields=schema_subset, max_results=10)]
print(results)
for i in results:
    print(dict(i))
BYTES_PER_GB = 2**30
hn_full.num_bytes / BYTES_PER_GB
def estimate_gigabytes_scanned(query, bq_client):
    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
    my_job_config = bigquery.job.QueryJobConfig()
    my_job_config.dry_run = True
    my_job = bq_client.query(query, job_config=my_job_config)
    BYTES_PER_GB = 2**30
    return my_job.total_bytes_processed / BYTES_PER_GB
estimate_gigabytes_scanned("SELECT Id FROM `bigquery-public-data.hacker_news.stories`", client)
estimate_gigabytes_scanned("SELECT * FROM `bigquery-public-data.hacker_news.stories`", client)