import pandas as pd
!pip install fastparquet
stackexchange_posts = pd.read_parquet('../input/get-stackexchange-archival-data/stackexchange_posts.parquet.gzip', engine='fastparquet')
stackexchange_posts.to_csv('stackexchange_posts.csv', index=False)