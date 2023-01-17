# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!apt-get install p7zip
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/test.tsv.7z
import os,sys
import pandas as pd
df_train = pd.read_csv('train.tsv', sep='\t')
df_test = pd.read_csv('test.tsv', sep='\t')
df_train.info()
df_train['name'].describe()
# Find most popular values
df_train['name'].value_counts().head(10)
# Case differences: Bundle and BUNDLE => lowercase & find most popular value 
df_train['name'].apply(lambda x: x.lower()).value_counts().head(10)
# Display length 
df_train['name'].astype(str).apply(len).hist()
# Find different names of similar products 
df_train[df_train['name'].apply(lambda x: x.lower().find('eyeshadow') >= 0)]
# Unique condition values
df_train['item_condition_id'].unique()
df_train['item_condition_id'].value_counts()
# Compute average price vs. item condition
df_train.groupby('item_condition_id')['price'].mean()
# Examples of ""
df_train[df_train['item_condition_id'] == 5].head()
df_train['category_name'].head()
df_train['category_name'].describe()
# How many items are missing a "category_name"
df_train['category_name'].isna().sum()
# PERCENTAGE of entries missing a "category_name" 
100.*df_train['category_name'].isna().sum()/df_train.shape[0]
# Split "category_name" by '/'
df_train['category_list'] = df_train['category_name'].astype(str).apply(lambda x: x.split('/'))
# How many levels are in the hierachy
df_train['category_list'].apply(len).value_counts()
# Create separate columns for category levels
df_train['level0'] = df_train['category_list'].apply(lambda x: x[0] if len(x) >= 1 else np.nan)
df_train['level1'] = df_train['category_list'].apply(lambda x: x[1] if len(x) >= 3 else np.nan)
df_train['level2'] = df_train['category_list'].apply(lambda x: x[2] if len(x) >= 3 else np.nan)
df_train['level3'] = df_train['category_list'].apply(lambda x: x[3] if len(x) >= 4 else np.nan)
df_train['level4'] = df_train['category_list'].apply(lambda x: x[4] if len(x) >= 5 else np.nan)
# print a number of unique categories in each level of the hierarchy
df_categories = df_train[['level0','level1','level2','level3','level4']].sort_values(['level0','level1','level2','level3','level4']).drop_duplicates()
for level in ['level0','level1','level2','level3','level4']:
    print(level,df_categories[level].nunique())
# Check for overlap between layers 0 and 1
set(df_categories['level0'].unique()).intersection(df_categories['level1'].unique())
# Intersect levels 0 and 2
set(df_categories['level0'].unique()).intersection(df_categories['level2'].unique())
# Intersect levels 1 and 2
set(df_categories['level1'].unique()).intersection(df_categories['level2'].unique())
df_train['brand_name'].describe()
df_train['brand_name'].value_counts()
# How many brands have 5 or more items
(df_train['brand_name'].value_counts() >= 5).sum()
df_train["price"].describe()
df_train["price"].hist()
# Plot price in the log domain
np.log(1+df_train["price"]).hist()
# price 0 or less
(df_train["price"] <= 0).sum()
df_train['shipping'].value_counts()
# Correlation between shipping and price
df_train.groupby('shipping')['price'].mean()
df_train['item_description'].value_counts()
# count missing descriptions
df_train['item_description'].isna().sum()
# Description length histogram
df_train['item_description'].astype(str).apply(lambda x: len(x)).hist()
# Logarithm of the description length histogram
np.log(1+df_train['item_description'].astype(str).apply(lambda x: len(x))).hist()
# Let's have a look of some extremely long descriptions
df_train['item_description'][df_train['item_description'].astype(str).apply(lambda x: len(x)) > 800].iloc[0]
