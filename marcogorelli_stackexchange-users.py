import pandas as pd
!pip install fastparquet
stackexchange_users = pd.read_parquet('../input/get-stackexchange-archival-data/stackexchange_users.parquet.gzip', engine='fastparquet')
stackexchange_users.to_csv('stackexchange_users.csv', index=False)