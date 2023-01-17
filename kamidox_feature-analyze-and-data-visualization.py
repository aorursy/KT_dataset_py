# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud



import string

import time



%matplotlib inline
df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_train.head()
print('Shape of train data set: ', df_train.shape)
df_train['item_description'].fillna(value="No description yet", inplace=True)
frac1 = 100 * df_train[df_train['price'] <= 0].shape[0] / df_train.shape[0]

print('%0.2f%% percent of product have 0 price. We may need drop them when we train our model.' % frac1)

df_train = df_train[df_train['price'] > 0]
df_train['price'].describe()
def price_hist(price, bins=100, r=[0,200], label='price', title='Price Distribution', **argv):

    plt.figure(figsize=(20, 15))

    plt.hist(price, bins=bins, range=r, label=label, **argv)

    plt.title(title, fontsize=15)

    plt.xlabel(label, fontsize=15)

    plt.ylabel('Samples', fontsize=15)

    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)

    plt.show()
price_hist(df_train['price'])
price_hist(np.log1p(df_train['price']), r=[0, 10])
free_shipping = df_train[df_train['shipping']==1]

print('%0.2f%% percent of the product with free shipping' % (100 * len(free_shipping)/len(df_train)))
def price_double_hist(price1, price2, label1='price 1', label2='price 2',

                      bins=100, r=[0,200], title='Double Price Distribution', **argv):

    plt.figure(figsize=(20, 15))

    plt.hist(price1, bins=bins, range=r, label=label1, **argv)

    plt.hist(price2, bins=bins, range=r, label=label2, **argv)

    plt.title(title, fontsize=15)

    plt.xlabel('Price', fontsize=15)

    plt.ylabel('Samples', fontsize=15)

    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)

    plt.show()
price_double_hist(price1=df_train[df_train['shipping']==1]['price'], 

                  price2=df_train[df_train['shipping']==0]['price'],

                  label1='Price with shipping',

                  label2='Price without shipping',

                  normed=True, alpha=0.6)
df = df_train[df_train['price']<100]

df.boxplot(column='price', by='item_condition_id', grid=True, figsize=(20, 15), return_type='dict');
start = time.time()

cloud = WordCloud(width=1440, height=1080).generate(" ".join(df_train['name'] + " " + df_train['item_description']))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

print('Time to compute and show word cloud: %0.2fs' % (time.time() - start))
df_train['has_description'] = 1

df_train.loc[df_train['item_description']=='No description yet', 'has_description'] = 0
price_double_hist(price1=df_train[df_train['has_description']==1]['price'], 

                  price2=df_train[df_train['has_description']==0]['price'],

                  label1='Price with description',

                  label2='Price without description',

                  normed=False, alpha=0.6)
price_double_hist(price1=df_train[df_train['has_description']==1]['price'], 

                  price2=df_train[df_train['has_description']==0]['price'],

                  label1='Price with description',

                  label2='Price without description',

                  normed=True, alpha=0.6)
df_train['has_price'] = 0

df_train.loc[df_train['item_description'].str.contains('[rm]', regex=False), 'has_price'] = 1

df_train.loc[df_train['name'].str.contains('[rm]', regex=False), 'has_price'] = 1
price_double_hist(price1=df_train[df_train['has_price']==0]['price'], 

                  price2=df_train[df_train['has_price']==1]['price'],

                  label1='Price without price in name/description',

                  label2='Price with price in name/description',

                  normed=False, alpha=0.6)
with_price = df_train[df_train['has_price']==1]

print('%0.2f%% percent of the product have price marked in name/description' % (100 * len(with_price)/len(df_train)))
price_double_hist(price1=df_train[df_train['has_price']==0]['price'], 

                  price2=df_train[df_train['has_price']==1]['price'],

                  label1='Price without price in name/description',

                  label2='Price with price in name/description',

                  normed=True, alpha=0.6)
tfidf = TfidfVectorizer(

                        min_df=2, lowercase =True,

                        analyzer='word', token_pattern=r'\w+', use_idf=True, 

                        smooth_idf=True, sublinear_tf=True, stop_words='english')



vect_tfidf = tfidf.fit_transform(df_train['name'] + " " + df_train['item_description'])
df_train['tfidf'] = vect_tfidf.sum(axis=1)
plt.figure(figsize=(20, 15))

plt.scatter(df_train['tfidf'], df_train['price'])

plt.title('Train price X name/item_description TF-IDF', fontsize=15)

plt.xlabel('Price', fontsize=15)

plt.ylabel('TF-IDF', fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15);
frac1 = 100 * df_train['category_name'].isnull().sum() / df_train.shape[0]

print('%0.2f percent empty category name' % frac1)
def transform_category_name(category_name):

    try:

        main, sub1, sub2 = category_name.split('/')

        return main, sub1, sub2

    except:

        return np.nan, np.nan, np.nan



df_train['category_main'], df_train['category_sub1'], df_train['category_sub2'] = zip(*df_train['category_name'].apply(transform_category_name))
frac1 = 100 * df_train['category_main'].isnull().sum() / df_train.shape[0]

print('%0.2f%% percent of the product do not fit 3 level category structure' % frac1)
main_categories = [c for c in df_train['category_main'].unique() if type(c)==str]

categories_sum=0

for c in main_categories:

    categories_sum+=100*len(df_train[df_train['category_main']==c])/len(df_train)

    print('{:<25}{:>10.4f}% of training data'.format(c, 100*len(df_train[df_train['category_main']==c])/len(df_train)))

print('{:<25}{:>10.4f}% of training data'.format('nan', 100-categories_sum))
df = df_train[df_train['price']<100]

df.boxplot(column='price', by='category_main', grid=True, figsize=(20, 15));
print('%d type of 2nd categories.' % len(df_train['category_sub1'].unique()))
def mean_price(groupby='category_sub1', cnt=20, top=True):

    df = df_train.groupby([groupby])['price'].agg(['size','sum'])

    df['mean_price']=df['sum']/df['size']

    df.sort_values(by=['mean_price'], ascending=(not top), inplace=True)

    df = df[:cnt]

    df.sort_values(by=['mean_price'], ascending=top, inplace=True)

    return df



def price_barh(df, title, ylabel):

    plt.figure(figsize=(20, 15))

    plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5)

    plt.yticks(range(0,len(df)), df.index, fontsize=15)

    plt.xticks(fontsize=15)

    plt.title(title, fontsize=15)

    plt.xlabel('Price', fontsize=15)

    plt.ylabel(ylabel, fontsize=15)
df = mean_price(cnt=50)

price_barh(df, 'highest mean price sorted by 2nd category', '2nd category')
df = mean_price(cnt=50, top=False)

price_barh(df, 'lowest mean price sorted by 2nd category', '2nd category')
print('%d type of 3rd categories.' % len(df_train['category_sub2'].unique()))
df = mean_price(cnt=50)

price_barh(df, 'highest mean price sorted by 3nd category', '3nd category')
df = mean_price(cnt=50, top=False)

price_barh(df, 'lowest mean price sorted by 3nd category', '3nd category')
brands = df_train['brand_name'].unique()

print('There are totaly %d brand names' % len(brands))
df = mean_price(groupby='brand_name', cnt=50, top=True)

price_barh(df, 'Most expensive product', 'brand')
df = mean_price(groupby='brand_name', cnt=50, top=False)

price_barh(df, 'Most cheap product', 'brand')
df_train['has_brand'] = 1

df_train.loc[df_train['brand_name'].isnull(), 'has_brand'] = 0



product_without_brand_name = df_train[df_train['has_brand']==0]

print('%0.4f%% percent of the product do not have brand name' % (100 * len(product_without_brand_name) / len(df_train)))
price_double_hist(price1=df_train[df_train['has_brand']==0]['price'], 

                  price2=df_train[df_train['has_brand']==1]['price'],

                  label1='Price without brand name',

                  label2='Price with brand name',

                  normed=False, alpha=0.6)
boundary = 10

below_boundary = df_train[df_train['price']<=boundary]

above_boundary = df_train[df_train['price']>boundary]



product_without_brand_name = below_boundary[below_boundary['has_brand']==0]

print('%0.4f%% percent of the product below price boundary(%dUSD) do not have brand name' % ((100 * len(product_without_brand_name) / len(below_boundary)), boundary))



product_without_brand_name = above_boundary[above_boundary['has_brand']==0]

print('%0.4f%% percent of the product above price boundary(%dUSD) do not have brand name' % ((100 * len(product_without_brand_name) / len(above_boundary)), boundary))
price_double_hist(price1=df_train[df_train['has_brand']==0]['price'], 

                  price2=df_train[df_train['has_brand']==1]['price'],

                  label1='Price without brand name',

                  label2='Price with brand name',

                  normed=True, alpha=0.6)
df = df_train[df_train['price'] > 100]

frac1 = 100 * df['brand_name'].isnull().sum() / df.shape[0]

print('There are still %0.4f%% percent of product do not have brand name with price above 100 USD' % frac1)
price_double_hist(price1=df[df['has_brand']==0]['price'], 

                  price2=df[df['has_brand']==1]['price'],

                  label1='Price without brand name for > $100',

                  label2='Price with brand name for > $100',

                  normed=False, alpha=0.6)