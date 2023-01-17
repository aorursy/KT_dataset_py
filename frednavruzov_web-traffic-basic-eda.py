# library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# garbage collector
import gc
# data import

# read train data
df_train = pd.read_csv('../input/train_1.csv', engine='c')
# check dtype distribution
print('dtype distribution:')
print(df_train.dtypes.value_counts())
# check df shape
print('\nTrain shape: {}'.format(df_train.shape))
# memory usage
print('Memory consumption, Mb: {:.2f}'.format(df_train.memory_usage().sum()/2**20))
# check sample output
df_train.head()
# reshape df to "flatten" form
df_train_flattened = pd.melt(df_train, id_vars='Page', var_name='date', value_name='traffic')
# see new shape and check it
print(df_train_flattened.shape)
assert df_train_flattened.shape[0] == (df_train.shape[1]-1)*df_train.shape[0], 'shit happened :('
# drop redundant df
del df_train
# check sample output
df_train_flattened.head()
# initial types/consumption
print(df_train_flattened.dtypes)
print('Memory consumption, Mb: {:.2f}'.format(df_train_flattened.memory_usage().sum()/2**20))
# add new column to distinquish between 0 and NaN in traffic
df_train_flattened['traffic_is_missing'] = df_train_flattened.traffic.isnull().astype(np.bool)
# now fillna as 0 and downcast dtype to np.int32, since max_value = 67264258
df_train_flattened.traffic.fillna(0, inplace=True)
df_train_flattened.traffic = df_train_flattened.traffic.astype(np.int32)
# convert datetime to datetime
df_train_flattened.date = pd.to_datetime(df_train_flattened.date, format="%Y-%m-%d")
gc.collect()
# optimized types/consumption
print(df_train_flattened.dtypes)
print('Memory consumption, Mb: {:.2f}'.format(df_train_flattened.memory_usage().sum()/2**20))
# create small dictionary df from 'Page' column
page_dict = pd.DataFrame({'Page': df_train_flattened.Page.unique()})
# split it to add some features
page_dict['agent_type'] = page_dict.Page.str.rsplit('_').str.get(-1)
page_dict['access_type'] = page_dict.Page.str.rsplit('_').str.get(-2)
page_dict['project'] = page_dict.Page.str.rsplit('_').str.get(-3)
# dirty hacking to get it :)
page_dict['page_name'] = page_dict.apply(
    lambda r: r['Page'][:-int(len(r['agent_type'])+len(r['access_type'])+len(r['project'])+3)], axis=1)

# add country
page_dict['source'] = page_dict.project.str.split('.').str.get(0).map(
    {'en':'English', 'ja':'Japanese', 
     'de':'German', 'fr':'France', 'zh':'Chinese',
    'ru':'Russian', 'es':'Spanish'}
).fillna('Other-Media')

# change dtypes to 'category'
for c in page_dict.columns:
    page_dict[c] = page_dict[c].astype('category')

# set index and see sample output
page_dict.set_index('Page', inplace=True)
page_dict.head()
# check unique values in categories:
print('User-agent')
print(page_dict.agent_type.value_counts())

print('\nProjects')
print(page_dict.project.value_counts())

print('\nAccess')
print(page_dict.access_type.value_counts())
# let's see general trend (averaged over pages by day)
df_train_flattened.groupby('date').aggregate({'traffic': 'mean'}).plot(
    figsize=(12,4), 
    title='Mean Page Views'
)
# let's see general trend (and median-based, more robust)
df_train_flattened.groupby('date').aggregate({'traffic': 'median'}).plot(
    figsize=(12,4), 
    title='Median Page Views'
)
# grouped plots, by source, mean pageviews (log-scaled for a better comparison)
# seems like in England and Russia August was hot :)
gc.collect()
df_train_flattened['source'] = df_train_flattened.Page.map(page_dict.source)
grouped = df_train_flattened.groupby(['source', 'date']).aggregate(
    {'traffic': 'mean'}).astype(np.float16)

fig = plt.figure(1, figsize=(12,8))
plt.ylabel('Views per Page')
plt.xlabel('Date')
plt.title('Average Pageviews for Different Sources')

for s in page_dict.source.unique():
    data = grouped.iloc[grouped.index.get_level_values('source') == s]\
    .reset_index(level=0, drop=True)
    plt.plot(data.index, data.traffic, label=s)

plt.legend()
plt.yscale('log') # logarithmic scale
pass
# to be continued...