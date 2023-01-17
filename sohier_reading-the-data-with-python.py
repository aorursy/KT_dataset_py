import pandas as pd
import pyarrow.parquet as pq
import os
os.listdir('../input')
train = pq.read_pandas('../input/train.parquet').to_pandas()
train.info()
train.columns[:5]
subset_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(5)]).to_pandas()
subset_train.info()
subset_train.head()
