# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import string

%matplotlib inline
df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')
df_train.head()
print('Train shape:{}\nTest shape:{}'.format(df_train.shape, df_test.shape))
plt.figure(figsize=(20, 15))
plt.hist(df_train['price'], bins=50, range=[0,250], label='price')
plt.title('Train "price" distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Samples', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
df_train['price'].describe()
plt.figure(figsize=(20, 15))
bins=50
plt.hist(df_train[df_train['shipping']==1]['price'], bins, normed=True, range=[0,250],
         alpha=0.6, label='price when shipping==1')
plt.hist(df_train[df_train['shipping']==0]['price'], bins, normed=True, range=[0,250],
         alpha=0.6, label='price when shipping==0')
plt.title('Train price over shipping type distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Normalized Samples', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
df = df_train[df_train['price']<100]

my_plot = []
for i in df_train['item_condition_id'].unique():
    my_plot.append(df[df['item_condition_id']==i]['price'])

fig, axes = plt.subplots(figsize=(20, 15))
bp = axes.boxplot(my_plot,vert=True,patch_artist=True,labels=range(1,6)) 

colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes.yaxis.grid(True)

plt.title('BoxPlot price X item_condition_id', fontsize=15)
plt.xlabel('item_condition_id', fontsize=15)
plt.ylabel('price', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

del df
cloud = WordCloud(width=1440, height=1080).generate(" ".join(df_train['item_description']
.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
df_train['has_description'] = 1
df_train.loc[df_train['item_description']=='No description yet', 'has_description'] = 0
plt.figure(figsize=(20, 15))
bins=50
plt.hist(df_train[df_train['has_description']==1]['price'], bins, range=[0,250],
         alpha=0.6, label='price when has_description==1')
plt.hist(df_train[df_train['has_description']==0]['price'], bins, range=[0,250],
         alpha=0.6, label='price when has_description==0')
plt.title('Train price X has_description type distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Samples', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
plt.figure(figsize=(20, 15))
bins=50
plt.hist(df_train[df_train['has_description']==1]['price'], bins, normed=True,range=[0,250],
         alpha=0.6, label='price when has_description==1')
plt.hist(df_train[df_train['has_description']==0]['price'], bins, normed=True,range=[0,250],
         alpha=0.6, label='price when has_description==0')
plt.title('Train price X has_description type distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Samples', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
def compute_tfidf(description):
    description = str(description)
    description.translate(string.punctuation)

    tfidf_sum=0
    words_count=0
    for w in description.lower().split():
        words_count += 1
        if w in tfidf_dict:
            tfidf_sum += tfidf_dict[w]
    
    if words_count > 0:
        return tfidf_sum/words_count
    else:
        return 0

tfidf = TfidfVectorizer(
    min_df=5, strip_accents='unicode', lowercase =True,
    analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
    smooth_idf=True, sublinear_tf=True, stop_words='english')
tfidf.fit_transform(df_train['item_description'].apply(str))
tfidf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
df_train['tfidf'] = df_train['item_description'].apply(compute_tfidf)
plt.figure(figsize=(20, 15))
plt.scatter(df_train['tfidf'], df_train['price'])
plt.title('Train price X item_description TF-IDF', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('TF-IDF', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
train_ds = pd.Series(df_train['item_description'].tolist()).astype(str)
test_ds = pd.Series(df_test['item_description'].tolist()).astype(str)

bins=100
plt.figure(figsize=(20, 15))
plt.hist(train_ds.apply(len), bins, range=[0,600], label='train')
plt.hist(test_ds.apply(len), bins, alpha=0.6,range=[0,600], label='test')
plt.title('Histogram of character count', fontsize=15)
plt.xlabel('Characters Number', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
bins=100
plt.figure(figsize=(20, 15))
plt.hist(train_ds.apply(lambda x: len(x.split())), bins, range=[0,100], label='train')
plt.hist(test_ds.apply(lambda x: len(x.split())), bins, alpha=0.6,range=[0,100], label='test')
plt.title('Histogram of word count', fontsize=15)
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
def transform_category_name(category_name):
    try:
        main, sub1, sub2= category_name.split('/')
        return main, sub1, sub2
    except:
        return np.nan, np.nan, np.nan

df_train['category_main'], df_train['category_sub1'], df_train['category_sub2'] = zip(*df_train['category_name'].apply(transform_category_name))
main_categories = [c for c in df_train['category_main'].unique() if type(c)==str]
categories_sum=0
for c in main_categories:
    categories_sum+=100*len(df_train[df_train['category_main']==c])/len(df_train)
    print('{:25}{:3f}% of training data'.format(c, 100*len(df_train[df_train['category_main']==c])/len(df_train)))
print('nan\t\t\t {:3f}% of training data'.format(100-categories_sum))
df = df_train[df_train['price']<80]

my_plot = []
for i in main_categories:
    my_plot.append(df[df['category_main']==i]['price'])
    
fig, axes = plt.subplots(figsize=(20, 15))
bp = axes.boxplot(my_plot,vert=True,patch_artist=True,labels=main_categories) 

colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']*2
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes.yaxis.grid(True)

plt.title('BoxPlot price X Main product category', fontsize=15)
plt.xlabel('Main Category', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
print('The data has {} unique 3rd level categories'.format(len(df_train['category_sub2'].unique())))

df = df_train.groupby(['category_sub2'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=False, inplace=True)
df = df[:20]
df.sort_values(by=['mean_price'], ascending=True, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5)
plt.yticks(range(0,len(df)), df.index, fontsize=15)
plt.xticks(fontsize=15)
plt.title('ASCENDING - 3rd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=15)
plt.show()
########################################################################

df = df_train.groupby(['category_sub2'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=True, inplace=True)
df = df[:50]
df.sort_values(by=['mean_price'], ascending=False, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5, color='r')
plt.yticks(range(0,len(df)), df.index, fontsize=15)
plt.xticks(fontsize=15)
plt.title('DESCENDING - 3rd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=15)
plt.show()
print('The data has {} unique 2nd level categories'.format(len(df_train['category_sub1'].unique())))

df = df_train.groupby(['category_sub1'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=False, inplace=True)
df = df[:20]
df.sort_values(by=['mean_price'], ascending=True, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5, color='green')
plt.yticks(range(0,len(df)), df.index, fontsize=15)
plt.xticks(fontsize=15)
plt.title('ASCENDING - 2nd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=15)
plt.show()
########################################################################

df = df_train.groupby(['category_sub1'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=True, inplace=True)
df = df[:50]
df.sort_values(by=['mean_price'], ascending=False, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', color='pink')
plt.yticks(range(0,len(df)), df.index, fontsize=15)
plt.xticks(fontsize=15)
plt.title('DESCENDING - 2nd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=15)
plt.show()