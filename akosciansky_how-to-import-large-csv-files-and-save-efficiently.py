import numpy as np 
import pandas as pd
import os
import tqdm
import gc
import feather
PATH = '../input/'
print(os.listdir("../input"))
%%time

with open(f'{PATH}NGS-2016-pre.csv') as file:
    n_rows = len(file.readlines())
print('2016 pre rows:', n_rows)

with open(f'{PATH}NGS-2016-reg-wk1-6.csv') as file:
    n_rows = len(file.readlines())
print('2016 wk1-6 rows:', n_rows)

with open(f'{PATH}NGS-2016-reg-wk7-12.csv') as file:
    n_rows = len(file.readlines())
print('2016 wk7-12 rows:', n_rows)

with open(f'{PATH}NGS-2016-reg-wk13-17.csv') as file:
    n_rows = len(file.readlines())
print('2016 wk13-17 rows:', n_rows)

with open(f'{PATH}NGS-2016-post.csv') as file:
    n_rows = len(file.readlines())
print('2016 post rows:', n_rows)

with open(f'{PATH}NGS-2017-pre.csv') as file:
    n_rows = len(file.readlines())
print('2017 pre rows:', n_rows)

with open(f'{PATH}NGS-2017-reg-wk1-6.csv') as file:
    n_rows = len(file.readlines())
print('2017 wk1-6 rows:', n_rows)

with open(f'{PATH}NGS-2017-reg-wk7-12.csv') as file:
    n_rows = len(file.readlines())
print('2017 wk7-12 rows:', n_rows)

with open(f'{PATH}NGS-2017-reg-wk13-17.csv') as file:
    n_rows = len(file.readlines())
print('2017 wk13-17 rows:', n_rows)

with open(f'{PATH}NGS-2017-post.csv') as file:
    n_rows = len(file.readlines())
print('2017 post rows:', n_rows)
# Only load the first 5 rows to get an idea of what the data look like
df_temp = pd.read_csv(f'{PATH}NGS-2016-pre.csv', nrows=5)
df_temp.head()
# Get information on the datatypes
df_temp.info()
# Find out the smallest data type possible for each numeric feature
float_cols = df_temp.select_dtypes(include=['float'])
int_cols = df_temp.select_dtypes(include=['int'])

for cols in float_cols.columns:
    df_temp[cols] = pd.to_numeric(df_temp[cols], downcast='float')
    
for cols in int_cols.columns:
    df_temp[cols] = pd.to_numeric(df_temp[cols], downcast='integer')

print(df_temp.info())
dtypes = {'Season_Year': 'int16',
         'GameKey': 'int8',
         'PlayID': 'int16',
         'GSISID': 'float32',
         'Time': 'str',
         'x': 'float32',
         'y': 'float32',
         'dis': 'float32',
         'o': 'float32',
         'dir': 'float32',
         'Event': 'str'}

col_names = list(dtypes.keys())
ngs_files = ['NGS-2016-pre.csv',
             'NGS-2016-reg-wk1-6.csv',
             'NGS-2016-reg-wk7-12.csv',
             'NGS-2016-reg-wk13-17.csv',
             'NGS-2016-post.csv',
             'NGS-2017-pre.csv',
             'NGS-2017-reg-wk1-6.csv',
             'NGS-2017-reg-wk7-12.csv',
             'NGS-2017-reg-wk13-17.csv',
             'NGS-2017-post.csv']
# Load each ngs file and append it to a list. 
# We will turn this into a DataFrame in the next step

df_list = []

for i in tqdm.tqdm(ngs_files):
    df = pd.read_csv(f'{PATH}'+i, usecols=col_names,dtype=dtypes)
    
    df_list.append(df)
# Merge all dataframes into one dataframe
ngs = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list
gc.collect()

# Convert Time to datetime
ngs['Time'] = pd.to_datetime(ngs['Time'], format='%Y-%m-%d %H:%M:%S')

# See what we have loaded
ngs.info()
# Turn Saeson_Year into a category and ultimately into an integer
ngs['Season_Year'] = ngs['Season_Year'].astype('category').cat.codes
# There are 2536 out of 66,492,490 cases where GSISID is NAN. Let's drop those to convert the data type
ngs = ngs[~ngs['GSISID'].isna()]
# Convert GSISID to integer
ngs['GSISID'] = ngs['GSISID'].astype('int32')
# Save to feather, which can then be loaded again - but much faster this time!
ngs.reset_index(drop=True).to_feather(f'ngs.feather')




