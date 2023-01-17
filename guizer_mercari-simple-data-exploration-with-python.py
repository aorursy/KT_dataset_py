import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import nltk



% matplotlib inline
sns.set(style='darkgrid')
def data_description(df):

    """

    Returns a dataframe with some informations about the variables of the input dataframe.

    """

    data = pd.DataFrame(index=df.columns)

    

    # the numeric data types

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



    for i in data.index:

        data.loc[i, 'count'] = df[i].count()

        data.loc[i, 'missing values'] = df[i].shape[0] - df[i].count()

        data.loc[i, 'unique values'] = len(df[i].unique())

        data.loc[i, 'type'] = df[i].dtypes

        

        # if the type is numeric compute statistical properties

        if df[i].dtypes in numerics: 

            data.loc[i, 'mean'] = df[i].mean()     

            data.loc[i, 'std'] = df[i].std()

            data.loc[i, 'min'] = df[i].min()

            data.loc[i, '25%'] = df[i].quantile(0.25)

            data.loc[i, 'median'] = df[i].quantile(0.5)

            data.loc[i, '75%'] = df[i].quantile(0.75)

            data.loc[i, 'max'] = df[i].max()

        else:

            count = df[i].str.count('[a-zA-Z]+')

            # mean, std, quartiles,  min and max of the number of words

            data.loc[i, 'mean_w'] = count.mean()

            data.loc[i, 'std_w'] = count.std()

            data.loc[i, 'min_w'] = count.min()

            data.loc[i, '25%_w'] = count.quantile(0.25)

            data.loc[i, 'median_w'] = count.quantile(0.5)

            data.loc[i, '75%_w'] = count.quantile(0.75)

            data.loc[i, 'max_w'] = count.max()

            

    return data.transpose()



def countplot(x, data, figsize=(10,5)):

    """

    Wraps the countplot function of seaborn and allow to specify the size of the figure.

    """ 

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.countplot(x=x, data=data, ax=ax, order=data[x].value_counts().index)

    for tick in ax.get_xticklabels():

        tick.set_rotation(90)

          

def subplots(x, y, z, data, hue=None, showfliers=False, figsize=(16,5)):

    """

    Boxplots and barplot. Wraps seabon's boxplot and barplot methods.

    """ 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    sns.barplot(x=x, y=y, data=data, order=data[x].value_counts().index, hue=hue, ax=ax1)

    sns.boxplot(x=x, y=y, data=data, order=data[x].value_counts().index, hue=hue, ax=ax2, showfliers=showfliers)

    for tick1, tick2 in zip(ax1.get_xticklabels(), ax2.get_xticklabels()):

        tick1.set_rotation(90)

        tick2.set_rotation(90)

        

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)    

    sns.barplot(x=x, y=z, data=data, order=data[x].value_counts().index, hue=hue, ax=ax1)

    sns.boxplot(x=x, y=z, data=data, order=data[x].value_counts().index, hue=hue, ax=ax2, showfliers=showfliers)

    for tick1, tick2 in zip(ax1.get_xticklabels(), ax2.get_xticklabels()):

        tick1.set_rotation(90)

        tick2.set_rotation(90)
# training set

train = pd.read_csv('../input/train.tsv', sep='\t')

print(train.shape)



# test set

test = pd.read_csv('../input/test.tsv', sep='\t')

print(test.shape)
train.sample(10)
data_description(train)
data_description(test)
countplot('item_condition_id', train, figsize=(8,4))
values = train['brand_name'].value_counts()

print(values)

countplot('brand_name', train[train['brand_name'].isin(values.index[0:50])] , figsize=(20,5))
plt.figure(figsize=(6,5))

sns.countplot(x='shipping', data=train)
# we plot the distribution distribution of price

plt.figure(figsize=(10,5))

sns.distplot(train['price'], kde=False)
# distribution of g = log(1+price)   (price=exp(g)-1)

# price = 0 <=> log(1+price) = 0

# this transformation might be useful later

plt.figure(figsize=(10,5))

train['log_price'] = np.log(train['price'] + 1)

sns.distplot(train['log_price'], kde=False)
train['category_name'].str.contains('/').fillna(False).value_counts()
test['category_name'].str.contains('/').fillna(False).value_counts()
# How many sub-categories in the training set

(train['category_name'].str.count('/')+1).value_counts()
# How many sub-categories in the test set

(test['category_name'].str.count('/')+1).value_counts()
# Extract the categories

train.loc[:, 'category_1'] = train['category_name'].map(lambda x: x.split('/')[0] if type(x) == type('a') and len(x.split('/')) > 0 else None)

train.loc[:, 'category_2'] = train['category_name'].map(lambda x: x.split('/')[1] if type(x) == type('a') and len(x.split('/')) > 1 else None)

train.loc[:, 'category_3'] = train['category_name'].map(lambda x: x.split('/')[2] if type(x) == type('a') and len(x.split('/')) > 2 else None)

train.loc[:, 'category_4'] = train['category_name'].map(lambda x: x.split('/')[3] if type(x) == type('a') and len(x.split('/')) > 3 else None)

train.loc[:, 'category_5'] = train['category_name'].map(lambda x: x.split('/')[4] if type(x) == type('a') and len(x.split('/')) > 4 else None)

    

print(train[['category_1','category_2','category_3','category_4','category_5']].count(axis=0))
countplot('category_1', train, figsize=(10,5))
countplot('category_2', train, figsize=(30,5))
values = train['category_3'].value_counts()

print(values)

countplot('category_3', train[train['category_3'].isin(values.index[0:50])] , figsize=(20,5))
countplot('category_4', train, figsize=(5,5))
countplot('category_5', train, figsize=(6,5))
# randomly print 10 names

for i in range(10):

    print(train['name'].sample(1).iloc[0])

    print()
# randomly print 10 descriptions

for i in range(10):

    print(train['item_description'].sample(1).iloc[0])

    print()
# Compute the number of occurences of each tag in a random sample of names

names_tags = nltk.pos_tag((train['name'].sample(10000) + ' ').sum())

names_tags_freq =  pd.Series(nltk.FreqDist(tag for (word, tag) in names_tags))
# Compute the number of occurence of each tag in a randm sample of item descriptions

descrip_tags = nltk.pos_tag((train['item_description'].sample(10000) + ' ').sum())

descrip_tags_freq =  pd.Series(nltk.FreqDist(tag for (word, tag) in descrip_tags))
# reoder the series

names_tags_freq = names_tags_freq.sort_values(ascending=False)

descrip_tags_freq = descrip_tags_freq.sort_values(ascending=False)
plt.figure(figsize=(15,5))

plt.bar(range(names_tags_freq.shape[0]), names_tags_freq, tick_label=names_tags_freq.index)
plt.figure(figsize=(15,5))

plt.bar(range(descrip_tags_freq.shape[0]), descrip_tags_freq, tick_label=descrip_tags_freq.index)
# Generate a word cloud image for name

wordcloud = WordCloud().generate((train['name'].sample(100000) + ' ').sum())

plt.figure(figsize=(10,5))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
# Generate a word cloud image for item description

wordcloud = WordCloud().generate((train['item_description'].sample(20000) + ' ').sum())

plt.figure(figsize=(10,5))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
subplots('item_condition_id', 'price', 'log_price', train, hue='shipping', showfliers=True)
results = train.groupby('brand_name').price.agg(['count','mean'])

results = results[results['count']>1000].sort_values(by='mean', ascending=False)

results.head(30)
values = train['brand_name'].value_counts().index[0:30]

subplots('brand_name', 'price', 'log_price', train[train['brand_name'].isin(values)], hue='shipping', 

         showfliers=True, figsize=(20,5))
subplots('shipping', 'price', 'log_price', train, figsize=(10,3))
subplots('category_1', 'price', 'log_price', train, hue='shipping', figsize=(16,5))
subplots('category_1', 'price', 'log_price', train, hue='item_condition_id', figsize=(20,5))
results = train.groupby('category_2').price.agg(['count','mean'])

results = results[results['count']>1000].sort_values(by='mean', ascending=False)

results.head(30)
values = train['category_2'].value_counts().index[0:30]

subplots('category_2', 'price', 'log_price', train[train['category_2'].isin(values)], figsize=(20,5))
values = train['category_3'].value_counts().index[0:30]

subplots('category_3', 'price', 'log_price', train[train['category_3'].isin(values)], figsize=(20,5))
subplots('category_4', 'price', 'log_price', train, figsize=(10,3))
subplots('category_5', 'price', 'log_price', train, figsize=(10,3))
fig, axes = plt.subplots(10, 2, figsize=(16,40))

cats = train['category_1'].value_counts().index

for i in range(10):

    ax1, ax2 = axes[i]    

    cat = cats[i]

    wordcloud1 = WordCloud().generate((train[train['category_1']==cat]['name'].sample(10000) + ' ').sum())

    wordcloud2 = WordCloud().generate((train[train['category_1']==cat]['item_description'].sample(10000) + ' ').sum())

    ax1.imshow(wordcloud1, interpolation='bilinear')

    ax2.imshow(wordcloud2, interpolation='bilinear')

    ax1.axis("off")

    ax2.axis("off")

    ax1.set_title('Most frequent words of name for category ' + cat)

    ax2.set_title('Most frequent words of the description for category ' + cat)