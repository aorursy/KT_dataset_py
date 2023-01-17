import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from bokeh.io import output_notebook, show

from bokeh.plotting import figure
train = pd.read_csv('../input/train.tsv', sep='\t', dtype={'item_condition_id':str, 'category_name':str})

test = pd.read_csv('../input/test.tsv', sep='\t', dtype={'item_condition_id':str, 'category_name':str})
train.drop(['train_id'], axis=1, inplace=True)

test_ids = test['test_id']

test.drop('test_id', axis=1, inplace=True)
print("Train shape:", train.shape)

print("Test shape:", test.shape)
assert list(train.drop('price', axis=1,).columns) == list(test.columns)
train.head()
train.info()
print("item_description = NaN in {} records".format(train[train.item_description.isnull()].shape[0]))

print("item_description = 'No description yet' in {} records".format(train[train.item_description == 'No description yet'].shape[0]))
def replace_text(df, variable, text_to_replace, replacement):

    df.loc[df[variable] == text_to_replace, variable] = replacement

    

replace_text(train, 'item_description', 'No description yet', '[ndy]')

replace_text(test, 'item_description', 'No description yet', '[ndy]')



train.loc[train['item_description'].isnull(), 'item_description'] = '[ndy]'

test.loc[train['item_description'].isnull(), 'item_description'] = '[ndy]'
train['item_description_nb_words'] = train['item_description'].str.split().apply(len)
output_notebook()

d = train['item_description_nb_words'].describe()



quartiles = ['Q1', 'Q2', 'Q3']

p = figure(x_axis_label='Nb words in item_description', y_axis_label='Quartile value', 

           x_range=quartiles, toolbar_location=None, tools="")

p.vbar(x=quartiles, top=[d['25%'],d['50%'],d['75%']], width=0.9, color='#EE9D31')

p.xgrid.grid_line_color = None

p.y_range.start = 0

show(p)
s = train['item_description_nb_words']



plt.figure(figsize=(12,8))

ax = sns.distplot(s, kde=False, bins=50)

ax.set(xlabel='Nb words in item_description', ylabel='Frequency')

ax.set(xticks=np.arange(0,s.max(),10))

plt.axvline(s.median(), color='r', linestyle='dashed', linewidth=1)  # vertical line at the median

yvals = ax.get_yticks()

ax.set_yticklabels(['{:,}'.format(y) for y in yvals])

plt.show();
train['name'].str.split().apply(len).describe()
train[~train.brand_name.isnull()]['brand_name'].str.split().apply(len).describe()
train.item_condition_id.value_counts(normalize=True).sort_index()
train[train.item_condition_id == '5'].head()
train[train.item_condition_id == '1'].head()
train.groupby('item_condition_id')['price'].describe()
train['nb_cat_slashes'] = train['category_name'].str.count('/')

train.nb_cat_slashes.value_counts(normalize=True).sort_index()
train[['cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5']] = train['category_name'].str.split('/', expand=True)
for c in ['cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5']:

    print('{} has {} unique values'.format(c, len(train[c].unique())))
train.cat_1.value_counts()
# Top 10 of cat_2

train.cat_2.value_counts()[:10]
# Top 10 of cat_3

train.cat_3.value_counts()[:10]
train.cat_4.value_counts()
train.cat_5.value_counts()
train[['cat_4','cat_5']].pivot_table(index='cat_4', columns='cat_5', aggfunc=len, fill_value=0)
train[['cat_1','cat_4']].pivot_table(index='cat_1', columns='cat_4', aggfunc=len, fill_value=0)
condition_counts = (train.groupby('item_condition_id')['cat_1']

                    .value_counts(normalize=True)

                    .rename('Percentage')

                    .reset_index())



plt.figure(figsize=(15,10))

sns.set(font_scale = 1.3)

ax = sns.barplot(x="cat_1", y="Percentage", hue="item_condition_id", data=condition_counts)

loc, labels = plt.xticks()

ax.set_xticklabels(labels, rotation=45)

plt.show();
train.shipping.mean()
train['price'].describe()
train[train.price == 0].shape[0]