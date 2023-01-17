from google.cloud import bigquery
client = bigquery.Client()
hn_dataset_ref = client.dataset('noaa_icoads', project='bigquery-public-data')
type(hn_dataset_ref)
hn_dset = client.get_dataset(hn_dataset_ref)
type(hn_dset)
[i.table_id for i in client.list_tables(hn_dset)]
icoads_core_1662_2000 = client.get_table(hn_dset.table('icoads_core_1662_2000'))
type(icoads_core_1662_2000)
[command for command in dir(icoads_core_1662_2000) if not command.startswith('_')]
icoads_core_1662_2000.schema
[type(i) for i in icoads_core_1662_2000.schema]
[i.name+", type: "+i.field_type for i in icoads_core_1662_2000.schema]
schema_subset = [col for col in icoads_core_1662_2000.schema if col.name in ('sea_surface_temp', 'sea_level_pressure', 'present_weather')]

results = [x for x in client.list_rows(icoads_core_1662_2000, start_index=100, selected_fields=schema_subset, max_results=10)]
print(results)
for i in results:

    print(dict(i))
BYTES_PER_GB = 2**30

icoads_core_1662_2000.num_bytes / BYTES_PER_GB
def estimate_gigabytes_scanned(query, bq_client):

    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun

    my_job_config = bigquery.job.QueryJobConfig()

    my_job_config.dry_run = True

    my_job = bq_client.query(query, job_config=my_job_config)

    BYTES_PER_GB = 2**30

    return my_job.total_bytes_processed / BYTES_PER_GB
estimate_gigabytes_scanned("SELECT sea_level_pressure FROM `bigquery-public-data.noaa_icoads.icoads_core_1662_2000`", client)
estimate_gigabytes_scanned("SELECT * FROM `bigquery-public-data.noaa_icoads.icoads_core_1662_2000`", client)