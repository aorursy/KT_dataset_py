import pandas as pd
!pip install fastparquet
stackexchange_postlinks = pd.read_parquet('../input/get-stackexchange-archival-data/stackexchange_postlinks.parquet.gzip', engine='fastparquet')
stackexchange_postlinks.to_csv('stackexchange_postlinks.csv', index=False)