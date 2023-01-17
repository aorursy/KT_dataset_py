import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/accepted_2007_to_2018q2.csv/accepted_2007_to_2018Q2.csv')
id_column_as_numeric = num_id = pd.to_numeric(df['id'], errors='coerce')
# We somehow have IDs which are not integers and get coerced into NaNs!
num_id = pd.to_numeric(df['id'], errors='coerce')
num_id.isna().sum()
# what are these non-numerical IDs?
df.loc[num_id.isna(), 'id']
# and lo, only the first column is defined, these must be summary statistic rows
# or rows that are follow-ups to the previous row
df.loc[num_id.isna(), :].head()
# how should we address this? just drop them!
print(f'df starts out with shape {df.shape}, but actually some rows are mal-formed...')
df = df.loc[num_id.notna(), :]
print(f'we dropped these mal-formed rows, and now we see that the df actually has an effective shape of {df.shape}.')