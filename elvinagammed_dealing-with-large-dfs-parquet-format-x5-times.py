import os



import pandas as pd

from pathlib import Path
%%time

datadir = Path('../input/zillow-prize-1')

train = pd.read_csv(datadir / 'properties_2017.csv')

# test = pd.read_csv(datadir / 'test.csv')

# submission = pd.read_csv(datadir / 'sample_submission.csv')
import pyarrow.parquet as pq
%%time



# --- save parquet format ---

outdir = Path('.')

os.makedirs(str(outdir), exist_ok=True)

train.to_parquet(outdir / 'train.parquet')