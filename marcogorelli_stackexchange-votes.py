import pandas as pd
!pip install fastparquet
stackexchange_votes = pd.read_parquet('../input/get-stackexchange-archival-data/stackexchange_votes.parquet.gzip', engine='fastparquet')
stackexchange_votes.to_csv('stackexchange_votes.csv', index=False)