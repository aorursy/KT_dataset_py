# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 50)



from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

train_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

test_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

sample_sub_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/sample_submission_V2.csv')
train_df.head()
test_df.head()
sample_sub_df.head()
# Function to reduce the memory usage

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)
train_df.isna().sum()
train_df = train_df[~train_df['winPlacePerc'].isna()]
print(f'Shape of training data: {train_df.shape}')

print(f'Shape of test data: {test_df.shape}')
plt.figure(figsize=(15,10))

plt.title("Target variable Distribution",fontsize=15)

sns.distplot(train_df['winPlacePerc'], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution of match duration (minutes)',fontsize=15)

sns.distplot(train_df['matchDuration']/60, kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title("Walking Distance Distribution",fontsize=15)

sns.distplot(train_df['walkDistance'][train_df['walkDistance'] < train_df['walkDistance'].quantile(.99)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title("Distribution of damage done",fontsize=15)

sns.distplot(train_df['damageDealt'][train_df['damageDealt'] < train_df['damageDealt'].quantile(.99)], kde=False)

plt.show()
len(train_df[(train_df['damageDealt'] == 0) & (train_df['winPlacePerc'] == 1)])
plt.figure(figsize=(15,10))

plt.title('Distribution of boosts taken during the match',fontsize=15)

sns.distplot(train_df['boosts'][train_df['boosts'] < train_df['boosts'].quantile(.999)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution # of heals taken during the match',fontsize=15)

sns.distplot(train_df['heals'][train_df['heals'] < train_df['heals'].quantile(.99)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution # of enemy players knocked down',fontsize=15)

sns.distplot(train_df['DBNOs'][train_df['DBNOs'] < train_df['DBNOs'].quantile(.999)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution # of kills',fontsize=15)

sns.distplot(train_df['kills'][train_df['kills'] < train_df['kills'].quantile(.999)], kde=False)

plt.show()
len(train_df[(train_df['kills'] == 0) & (train_df['winPlacePerc'] == 1)])
plt.figure(figsize=(15,10))

plt.title('Distribution of # of team kills (Friendly fire)',fontsize=15)

sns.distplot(train_df['teamKills'], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution of # of revives',fontsize=15)

sns.distplot(train_df['revives'][train_df['revives'] < train_df['revives'].quantile(.9999)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution of # of weapons acquired',fontsize=15)

sns.distplot(train_df['weaponsAcquired'][train_df['weaponsAcquired'] < train_df['weaponsAcquired'].quantile(.9999)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

ax = sns.barplot(train_df.groupby(['matchType']).size().reset_index(name='counts')['matchType'], train_df.groupby(['matchType']).size().reset_index(name='counts')['counts'])

ax.set(xlabel='Match Type', ylabel='# of records', title='Match Type vs. # of records')

ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution of # of assists',fontsize=15)

sns.distplot(train_df['assists'][train_df['assists'] < train_df['assists'].quantile(.9999)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution of ride distance',fontsize=15)

sns.distplot(train_df['rideDistance'][train_df['rideDistance'] < train_df['rideDistance'].quantile(.9)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution of walk distance',fontsize=15)

sns.distplot(train_df['walkDistance'][train_df['walkDistance'] < train_df['walkDistance'].quantile(.99)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution of swim distance',fontsize=15)

sns.distplot(train_df['swimDistance'][train_df['swimDistance'] < train_df['swimDistance'].quantile(.99)], kde=False)

plt.show()
plt.figure(figsize=(15,10))

plt.title('Distribution of # of headshot kills',fontsize=15)

sns.distplot(train_df['headshotKills'][train_df['headshotKills'] < train_df['headshotKills'].quantile(.999)], kde=False)

plt.show()