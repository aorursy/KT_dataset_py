import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
train = pd.read_table("../input/train.tsv")
train.head()
train.info()
train.describe()
#sns.heatmap(train.isnull(), cmap = 'viridis')
def norm_cat_str(s):

    

    try:

        if len(s.split('/')) == 3:

            return s + '/ / '

        elif len(s.split('/')) == 4:

            return s +'/ '

        else:

            return s

    except:

        return '/ / / / '
train['category_name'] = train['category_name'].apply(norm_cat_str)
train.loc[train['brand_name'].isnull(),'brand_name'] = 'Brand not specified'
train.head()
#sns.heatmap(train.isnull(), cmap = 'viridis')
train['cat1'] = train['category_name'].apply(lambda x: x.split('/')[0])
train['cat2'] = train['category_name'].apply(lambda x: x.split('/')[1])

train['cat3'] = train['category_name'].apply(lambda x: x.split('/')[2])

#train['cat4'] = train['category_name'].apply(lambda x: x.split('/')[3])

#train['cat5'] = train['category_name'].apply(lambda x: x.split('/')[4])
train.head()
train[train['brand_name'].isnull()].shape
train.drop('category_name', axis = 1, inplace = True)
train.head(2)
train['name'].nunique() / train.shape[0]

#value_counts()[0:10]
train['brand_name'].value_counts()[0:20]
train.columns
sns.countplot(x = 'item_condition_id', data = train)
df = train['brand_name'].value_counts()[0:20]

df

brand_count_df = pd.DataFrame(df)

brand_count_df['brand'] = brand_count_df.index

brand_count_df
brand_count_df.columns
train.columns
plt.figure(figsize = (18,6))

sns.barplot(x = 'brand', y = 'brand_name', data = brand_count_df.drop('Brand not specified', axis = 0))

plt.xticks(rotation = 'vertical')
brand_mean_price = train.groupby(by = 'brand_name', axis =0).mean()['price'].sort_values(ascending = False)

brand_mean_price[0:20]
cat1_mean_price = train.groupby(by = 'cat1', axis =0).mean()['price'].sort_values(ascending = False)

cat1_mean_price
#train.groupby(by = 'cat2', axis =0).mean()['price'].sort_values(ascending = False)

cat2_mean_price = train.groupby(by = 'cat2', axis =0).mean()['price'].sort_values(ascending = False)
cat2_mean_price = cat2_mean_price.to_frame('Avg_Price')

cat2_mean_price['cat2'] = cat2_mean_price.index
top_cat2 = cat2_mean_price['cat2'][0:10]

top_cat2[0:10]
sns.boxplot(x = 'cat1', y = 'price', data = train)

plt.xticks(rotation = 'vertical')
cat2_df = train.loc[~train['cat2'].isin(top_cat2)]

plt.figure(figsize = (10,6))

sns.boxplot(x = 'cat2', y = 'price', data = cat2_df)

plt.xticks(rotation = 'vertical')

sns.countplot(x = 'shipping', data = train)
#To be continued..