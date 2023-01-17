import pandas as pd

import dask

from dask.distributed import Client, progress

from dask import delayed

client = Client()

client



dask.__version__
def make_df(inst_sample):

    cols = ['one', 'two', 'three']

    new_cols = [x + inst_sample for x in cols]

    df = pd.DataFrame(1, index=['a', 'b', 'c'], columns=new_cols)

    return df
all_samples = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
df_list = []

for inst_sample in all_samples:

    inst_df = make_df(inst_sample)

    df_list.append(inst_df)
df_merge = pd.concat(df_list, axis=1)

print('df_merge shape', df_merge.shape)
df_list = []

for inst_sample in all_samples:

    inst_df = delayed(make_df)(inst_sample)

    df_list.append(inst_df)
df_merge = delayed(pd.concat)(df_list, axis=1).compute()

print(df_merge.shape)
df_merge