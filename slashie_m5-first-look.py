import os

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
files = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files[filename[:4]] = os.path.join(dirname, filename)

files
%%time

train_df, cal_df, prc_df, sub_df = [pd.read_csv(files[f]) for f in ['sale', 'cale', 'sell', 'samp']]
train_df.sample(5)
print("There are %d unique item ids to forecast!"%train_df.shape[0])
n = 100 # number of items to sample

sales = train_df[[c for c in train_df.columns if c.startswith('d_')]].sample(n)

fig, ax = plt.subplots(1, 1, facecolor='w', figsize=(15,10))

ax = sns.heatmap(sales>0, cbar=False, xticklabels=False, yticklabels=False, cmap="GnBu")

plt.title("Heatmap of >0 sales indicator for %d randomly selected items"%n, fontsize=16)

plt.ylabel("Items")

plt.xlabel("Time")

plt.show()
sub_df['type'] = sub_df['id'].apply(lambda x: x.split('_')[-1]).astype('category')

sub_df['type'].value_counts()
val_df = sub_df[sub_df['type']=='validation'].drop('type',1)

val_df.sample(5)
print("First 3 rows of the calendar data:")

display(cal_df.head(3))

print("Last 3 rows of the calendar data:")

display(cal_df.tail(3))
prc_df.loc[:,'id'] = prc_df['item_id'] + '_' + prc_df['store_id'] + '_validation'

prc_df = prc_df.drop(['store_id', 'item_id'],1)

display(prc_df.sample(5))

print("Earliest week: %d"%prc_df['wm_yr_wk'].min())

print("Latest week: %d"%prc_df['wm_yr_wk'].max())

print("Number of unique id: %d"%len(prc_df['id'].unique()))
id_grp = prc_df.groupby('id')

min_wk = id_grp['wm_yr_wk'].min()

max_wk = id_grp['wm_yr_wk'].max()

print("Summary statistics for latest week of prices for each item shows that ALL items have prices available in the last week:")

display(max_wk.describe())

print("New items appear to be rolled out in a staggered, seasonal fashion:")

fig, ax = plt.subplots(1, 1, facecolor='w', figsize=(12,6))

ax.hist(min_wk, bins=25)

plt.title('Minimum week across different item IDs', fontsize=16)

plt.show()