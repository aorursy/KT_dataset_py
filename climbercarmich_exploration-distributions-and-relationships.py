import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from dataninja import kick
import altair as alt
import seaborn as sns
import cufflinks as cf
import plotly.offline as plotly
%matplotlib inline

cf.go_offline()
%%time
dtypes = {'user_type':'category',
          'category_name':'category',
          'parent_category_name':'category',
          'region':'category',
          'city':'category'}

train = pd.read_csv("../input/train.csv",
                    parse_dates=['activation_date'],
                    dtype=dtypes)
test = pd.read_csv("../input/test.csv",
                   parse_dates=['activation_date'],
                   dtype=dtypes)

target = 'deal_probability'
all_data = pd.concat([train.assign(dataset='train'),test.assign(dataset='test')],sort=False).reset_index(drop=True)
dataset_colors = ['red','orange']
(all_data
 .groupby('dataset')
 .nunique()
 .unstack()
 .sort_values(ascending=False)
 .unstack()
 .sort_values('test',ascending=False)
 .iloc[8:]
 .drop([target]+['dataset'])
 .iplot(title='Unique Value Counts for Categoricals by Dataset',kind='bar',colors=dataset_colors)
)
#features with missing values
missing_vals = all_data.isnull().sum()
missing_vals = missing_vals[missing_vals>0]
missing_vals = missing_vals.index.tolist()
missing_vals.remove(target)

obs_counts = all_data.groupby('dataset').size().reset_index(name='obs_count')

(all_data
 .groupby('dataset')
 .apply(lambda x: x.isnull().sum())
 .merge(obs_counts,left_index=True,right_on='dataset')
 .set_index('dataset')
 .loc[:,missing_vals+['obs_count']]
 .transform(lambda x: x/x.max(),axis=1)
 .drop('obs_count',axis=1)
 .T
 .iplot(kind='bar',title='Missing Values Comparison - Train vs. Test',colors=dataset_colors)
)
(all_data
 .set_index('dataset')
 .loc[:,missing_vals]
 .isnull().sum(axis=1)
 .reset_index(name='item_missing_val_count')
 .groupby(['dataset','item_missing_val_count'])
 .size().reset_index(name='item_missing_val_count_count')
 .merge(obs_counts,on='dataset')
 .assign(item_missing_val_count_scaled = lambda x: x.item_missing_val_count_count/x.obs_count)
 .set_index(['dataset','item_missing_val_count']).item_missing_val_count_scaled
 .unstack(0)
 .iplot(title='Missing Values per Item Comparisons',kind='bar',colors=dataset_colors)
)
train.groupby('user_type')[target].mean().sort_values().iplot(mode='markers+lines',title='Average Deal Probability by User Type',color='blue')
train.groupby('parent_category_name')[target].mean().sort_values().iplot(mode='markers+lines',title='Average Deal Probability by Parent Category',color='green')
train.groupby('category_name')[target].mean().sort_values().iplot(mode='markers+lines',title='Average Deal Probability by Category',color='gold',margin={'b':120})
train.groupby('region')[target].mean().sort_values().iplot(mode='markers+lines',title='Average Deal Probability by Region',color='magenta')
train.query('price<5000000').groupby('price')[target].mean().rolling(1000,min_periods=50).mean().iplot(title='Deal Probabilty by Price',color='purple')
train.groupby('image_top_1')[target].mean().rolling(1000,min_periods=50).mean().iplot(title='Deal Probabilty by Image Top 1',color='black')
all_data.groupby(['dataset','activation_date']).size().unstack(0).iplot(title='Train vs. Test by Activation Date',colors=dataset_colors)
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(data = all_data
            .assign(weekday = lambda x: x.activation_date.dt.weekday,
                    week = lambda x: x.activation_date.dt.week)
            .groupby(['weekday','week'])
            .size()
            .unstack(0),
            cmap='viridis',
            ax=ax
            )
plt.title('Listings by Week and Day');
all_data.description.str.len().hist(bins=50,color='black',figsize=(10,6))
plt.title('Description Length Distribution');
all_data.title.str.len().hist(bins=50,color='gold',figsize=(10,6))
plt.title('Title Length Distribution');
train[target].iplot(kind='hist',bins=20,title='Deal Probability Distribution')
train[target].value_counts().transform(lambda x: x/x.sum()).head(10).round(2)