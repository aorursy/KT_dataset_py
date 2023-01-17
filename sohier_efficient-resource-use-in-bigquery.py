from google.cloud import bigquery
client = bigquery.Client()
github_dset = client.get_dataset(client.dataset('github_repos', project='bigquery-public-data'))
commits_table = client.get_table(github_dset.table('commits'))
BYTES_PER_GB = 2**30
print(f'The commits table is {int(commits_table.num_bytes/BYTES_PER_GB)} GB')
commits_header = [x for x in client.list_rows(commits_table, max_results=3)]
print('\n'.join([str(dict(i)) for i in commits_header]))
def estimate_gigabytes_scanned(query, bq_client):
    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
    my_job_config = bigquery.job.QueryJobConfig()
    my_job_config.dry_run = True
    my_job = client.query(query, job_config=my_job_config)
    BYTES_PER_GB = 2**30
    return my_job.total_bytes_processed / BYTES_PER_GB
estimate_gigabytes_scanned("SELECT * FROM `bigquery-public-data.github_repos.commits`", client)
estimate_gigabytes_scanned("SELECT commit, author.date FROM `bigquery-public-data.github_repos.commits`", client)
estimate_gigabytes_scanned("SELECT * FROM `bigquery-public-data.github_repos.commits` LIMIT 10", client)
QUERY = """SELECT COUNT(commit) FROM `bigquery-public-data.github_repos.commits`
        WHERE author.date = TIMESTAMP("2014-12-25")"""
estimate_gigabytes_scanned(QUERY, client)
