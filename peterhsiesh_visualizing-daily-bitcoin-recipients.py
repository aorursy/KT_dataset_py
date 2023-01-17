import os
import pandas as pd
from google.cloud import bigquery
from datetime import datetime,timezone
# from bq_helper import BigQueryHelper
# SERVICE_ACCOUNT_JSON = "file/Bitcoin-Tracing-472072344e9c.json"
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_JSON
client = bigquery.Client()

def list_field(dataset):
#list all the fields and sub-fields in dataset
    hn_dataset_ref = client.dataset(dataset[1], project=dataset[0])
    hn_dset = client.get_dataset(hn_dataset_ref)
    for t in client.list_tables(hn_dset):
        hn_full = client.get_table(hn_dset.table(t.table_id))
        for f1 in hn_full.schema:
            if (f1.fields):
                for f2 in f1.fields:
                    print("".join([t.table_id, ":",f1.name, "[",f1.field_type,"]",":",f2.name, "[",f2.field_type,"]"]))
            else:
                print("".join([t.table_id, ":",f1.name, "[",f1.field_type,"]"]))


if __name__ == "__main__":
    job_config = bigquery.QueryJobConfig()
#     job_config.dry_run = True
#     job_config.use_query_cache = False
#     list_field(['bigquery-public-data', 'bitcoin_blockchain'])
#     exit
    date_start_str = "2016-1-30"
    date_stop_str  = "2016-1-31"
    dt = datetime.strptime(date_start_str, "%Y-%m-%d")
    timestamp_start = int(dt.replace(tzinfo=timezone.utc).timestamp())*1000 # to millisecond
    dt = datetime.strptime(date_stop_str, "%Y-%m-%d")
    timestamp_stop  = int(dt.replace(tzinfo=timezone.utc).timestamp())*1000 # to millisecond
    
    query = """
    SELECT
        timestamp,o.output_pubkey_base58
    FROM
        `bigquery-public-data.bitcoin_blockchain.transactions`,UNNEST(outputs) AS o
    WHERE
        timestamp > """ + str(timestamp_start) + """ AND timestamp < """ + str(timestamp_stop) + """
    ORDER BY
       timestamp
    """
    
    print("Performing query :\n" + query)
    query_job = client.query(query,job_config=job_config)
 
#     assert query_job.state == 'DONE'
#     assert query_job.dry_run
    
    if (query_job.total_bytes_processed):
        n = query_job.total_bytes_processed;
    else:
        n = 0
    print("This query will process {} bytes.".format(n/(1024^3)))
    
    if (job_config.dry_run):
        exit
    else:
        iterator = query_job.result(timeout=30)
        rows = list(iterator)
        transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
#         print(transactions.head(10))
        print(len(transactions))