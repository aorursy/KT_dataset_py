from bq_helper import BigQueryHelper

bq = BigQueryHelper("cloud-training-demos", "taxifare_kaggle")
bq.list_tables()