import pandas as pd
!pip install fastparquet
stackoverflow_posts = pd.read_parquet('../input/get-stackexchange-archival-data/stackoverflow_posts.parquet.gzip', engine='fastparquet')
stackoverflow_posts.to_csv('stackoverflow_posts.csv', index=False)