import os

from sklearn.model_selection import StratifiedKFold

import pandas as pd

import matplotlib.pyplot as plt

from itertools import islice



# get train csv as pandas dataframe

df = pd.read_csv(os.path.join('../input/global-wheat-detection', 'train.csv'))



# get a df with just image and source columns

# such that dropping duplicates will only keep unique image_ids

image_source = df[['image_id', 'source']].drop_duplicates()



# get lists for image_ids and sources

image_ids = image_source['image_id'].to_numpy()

sources = image_source['source'].to_numpy()



# do the split

# in other words:

# split up our data into 10 buckets making sure that each bucket

#  has a more or less even distribution of sources

# Note the use of random_state=1 to ensure the split is the same each time we run this code

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

split = skf.split(image_ids, sources) # second arguement is what we are stratifying by



# so now `split` is an iterator

# each iteration gives us a set of indices pointing to the train rows of the df 

#  (in this case 90% of all the data)

#  and a set of indices pointing to the val rows of the df

#  (in this case 10% of the data)

# we can use islice to control which split we select

select = 0

train_ix, val_ix = next(islice(split, select, select+1))



# translate indices to ids

train_ids = image_ids[train_ix]

val_ids = image_ids[val_ix]



# create corresponding dfs

train_df = df[df['image_id'].isin(train_ids)]

val_df = df[df['image_id'].isin(val_ids)]
print(f'# train images: {train_ids.shape[0]}')

print(f'# val images: {val_ids.shape[0]}')



fig = plt.figure(figsize=(20, 5))

counts = train_df['source'].value_counts()

ax1 = fig.add_subplot(1,2,1)

a = ax1.bar(counts.index, counts)

counts = val_df['source'].value_counts()

ax2 = fig.add_subplot(1,2,2)

a = ax2.bar(counts.index, counts)