import nltk

import string

import re

import numpy as np

import pandas as pd

import pickle

#import lda



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")



from nltk.stem.porter import *

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from sklearn.feature_extraction import stop_words



from collections import Counter

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

%matplotlib inline



import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool

from bokeh.models import ColumnDataSource

from bokeh.plotting import figure, show, output_notebook

#from bokeh.transform import factor_cmap



import warnings

warnings.filterwarnings('ignore')

import logging

logging.getLogger("lda").setLevel(logging.WARNING)



PATH = "../input/"
len(stopwords.words('english'))
len(stop_words.ENGLISH_STOP_WORDS)
train = pd.read_csv(f"{PATH}train.tsv",sep='\t')

test = pd.read_csv(f"{PATH}test.tsv",sep='\t')
print(train.shape)

print(test.shape)
frac = 0.1

train = train.sample(frac=frac,random_state=200)

test = test.sample(frac=frac,random_state=200)
print(train.shape)

print(test.shape)
train.dtypes
train.head()
train['price'].describe()
fig,ax = plt.subplots(1,2,figsize=[10,5])

ax = plt.subplot(1,2,1)

sns.distplot(train['price'],hist=False,ax=ax,bins=50)

ax.set(ylabel='Density',title='Distribution of price')

ax = plt.subplot(1,2,2)

sns.distplot(np.log1p(train['price']),hist=False,ax=ax,bins=50)

ax.set(xlabel='log1p(price)',ylabel='Density',title='Distribution of log1p(price)')
train['shipping'].value_counts()
shipping_buyer_price = train.loc[train['shipping'] == 0,'price']

shipping_seller_price = train.loc[train['shipping'] == 1,'price']



fig = plt.figure(figsize=[10,5])

sns.distplot(np.log1p(shipping_buyer_price),bins=50,kde=False,color='red',label='buyer')

sns.distplot(np.log1p(shipping_seller_price),bins=50,kde=False,color='blue',label='seller')

plt.xlabel('log1p(price)')

plt.legend()
train['category_name'].nunique()
train['category_name'].isnull().sum()
category_null = train.loc[train['category_name'].isnull()]

category_notnull = train.loc[train['category_name'].notnull()]



# print(category_notnull.shape)

# category_notnull.info()



sep_categories = category_notnull['category_name'].apply(lambda x:x.split('/'))



col_names = ['general_cat','subcat_1','subcat_2']



cats = []



for cats_ in sep_categories:

    cat_dict = {col_names[0]:cats_[0],col_names[1]:cats_[1],col_names[2]:cats_[2]}

    cats.append(cat_dict)

    

df_cats = pd.DataFrame(cats)



category_notnull = category_notnull.reset_index(drop=True)

category_notnull = pd.concat([category_notnull,df_cats],axis=1)



for col in col_names:

    category_null[col] = 'No Label'



# assert category_notnull.index.max() == category_notnull.train_id.max()



train = pd.concat([category_notnull,category_null],axis=0,sort=False).sort_values(by='train_id').reset_index(drop=True)

train.info()
# # reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55

# def split_cat(text):

#     try: return text.split("/")

#     except: return ("No Label", "No Label", "No Label")
# train['general_cat'], train['subcat_1'], train['subcat_2'] = \

# zip(*train['category_name'].apply(lambda x: split_cat(x)))

# train.head()
print("There are %d unique general-categories." % train['general_cat'].nunique())
print("There are %d unique first sub-categories." % train['subcat_1'].nunique())
print("There are %d unique second sub-categories." % train['subcat_2'].nunique())
train['general_cat']
plt.figure(figsize=[8,6])

train['general_cat'].value_counts().plot(kind='bar',linewidth=2)

plt.title('Number of Items by Main Category')

plt.ylabel('Count')

plt.xlabel('Categories')

plt.xticks(rotation=60)
train['subcat_1'].value_counts()[:15].plot(kind='bar',linewidth=2, figsize=[12,10])

plt.title('Number of Items by Subset_1')

plt.ylabel('Count')

plt.xlabel('Categories')

plt.xticks(rotation=60)
fig = plt.figure(figsize=[12,10])

sns.boxplot(x=train['general_cat'],y=np.log1p(train['price']))

plt.xticks(rotation=60)
def wordCount(text):

    # convert to lower case and strip regex

    try:

         # convert to lower case and strip regex

        text = text.lower()

        #regular expression pattern 생성

        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')

        #패턴을 이용하여 들어오는 텍스트에서 해당 패턴의 기호들을 제거하는 것

        txt = regex.sub(" ", text)

        # tokenize

        # words = nltk.word_tokenize(clean_txt)

        # remove words in stop words

        words = [w for w in txt.split(" ") \

                 if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3]

        return len(words)

    except: 

        return 0
train['item_description'][3]
regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
regex.sub(" ",train['item_description'][3])
list(stop_words.ENGLISH_STOP_WORDS)[:5]
# add a column of word counts to both the training and test set

train['desc_len'] = train['item_description'].apply(lambda x: wordCount(x))

test['desc_len'] = test['item_description'].apply(lambda x: wordCount(x))
train.head()
df = train.groupby('desc_len')['price'].mean().reset_index()
df.columns
plt.figure(figsize=[20,8])

sns.pointplot(x=df.columns[0],y=df.columns[1],data=df)

plt.xticks(rotation=60)
train['item_description'].isnull().sum()
train = train.loc[train['item_description'].notnull()]
stop = set(stopwords.words('english'))

def tokenize(text):

    """

    sent_tokenize(): segment text into sentences

    word_tokenize(): break sentences into words

    """

    try: 

        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')

        text = regex.sub(" ", text) # remove punctuation

        

        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]



        tokens = []

        for token_by_sent in tokens_:

            tokens += token_by_sent



        tokens = list(filter(lambda t: t.lower() not in stop, tokens))

        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]

        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]

        

        return filtered_tokens

            

    except TypeError as e: print(text,e)
cat_desc = dict()

for cat in train.general_cat.unique(): 

    text = " ".join(train.loc[train['general_cat']==cat, 'item_description'].values)

    cat_desc[cat] = tokenize(text)
# flat list of all words combined

flat_lst = [item for sublist in list(cat_desc.values()) for item in sublist]

allWordsCount = Counter(flat_lst)

all_top10 = allWordsCount.most_common(20)

x = [w[0] for w in all_top10]

y = [w[1] for w in all_top10]
len(flat_lst)
len(allWordsCount)
all_top10
# stop = set(stopwords.words('english'))

# def tokenize(text):

#     """

#     sent_tokenize(): segment text into sentences

#     word_tokenize(): break sentences into words

#     """

#     try: 

#         regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')

#         text = regex.sub(" ", text) # remove punctuation

        

# #         print(sent_tokenize(text))

        

#         tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]

# #         print(tokens_)

#         tokens = []

#         for token_by_sent in tokens_:

#             tokens += token_by_sent

# #         print(tokens)

#         tokens = list(filter(lambda t: t.lower() not in stop, tokens))

# #         print(tokens)

#         filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]

# #         print(filtered_tokens)

#         filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]

# #         print(filtered_tokens)

        

#         return filtered_tokens

            

#     except TypeError as e: print(text,e)
train['tokens'] = train['item_description'].apply(tokenize)

test['tokens'] = train['item_description'].apply(tokenize)
for description, tokens in zip(train['item_description'].head(),

                              train['tokens'].head()):

    print('description:', description)

    print('tokens:', tokens)

    print()
#어짜피 토큰 만들어놨는데 다시 조인해서 토크나이즈 하는 것은 낭비라고 생각

cat_desc = dict()

for cat in train.general_cat.unique():

    cat_list = []

    for token in train.loc[train['general_cat']==cat,'tokens']:

        for one in token:

            cat_list.append(one)

    cat_desc[cat] = cat_list
women100 = Counter(cat_desc['Women']).most_common(100)

beauty100 = Counter(cat_desc['Beauty']).most_common(100)

kids100 = Counter(cat_desc['Kids']).most_common(100)

electronics100 = Counter(cat_desc['Electronics']).most_common(100)
women100
def generate_wordcloud(tup):

    wordcloud = WordCloud(background_color='white',max_words=50,max_font_size=40,random_state=42).generate(str(tup))

    return wordcloud
fig,ax = plt.subplots(2,2,figsize=[30,15])



ax = plt.subplot(2,2,1)

ax.imshow(generate_wordcloud(women100),interpolation="bilinear")

ax.axis('off')



ax = plt.subplot(2,2,2)

ax.imshow(generate_wordcloud(beauty100))

ax.axis('off')



ax = plt.subplot(2,2,3)

ax.imshow(generate_wordcloud(kids100))

ax.axis('off')



ax = plt.subplot(2,2,4)

ax.imshow(generate_wordcloud(electronics100))

ax.axis('off')
train.head()
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(min_df=10,max_features=180000,tokenizer=tokenize,ngram_range=(1,2))
all_desc = np.append(train['item_description'].values,test['item_description'].values)

vz = vectorizer.fit_transform(list(all_desc))
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

tfidf = pd.DataFrame(columns=['tfidf']).from_dict(

                    dict(tfidf), orient='index')

tfidf.columns = ['tfidf']
tfidf.sort_values(by='tfidf',ascending=True).head(10)
tfidf.sort_values(by='tfidf',ascending=False).head(10)
trn = train.copy()

tst = test.copy()

trn['is_train'] = 1

tst['is_train'] = 0



#t-SNE가 시간이 많이 걸리는 작업이기 때문에 임의의 샘플사이즈로 나누고 이를 차원분할해보자

sample_sz = 15000



combined_df = pd.concat([trn,tst])

combined_sample = combined_df.sample(n=sample_sz)

vz_sample = vectorizer.fit_transform(list(combined_sample['item_description']))
vz_sample.shape
from sklearn.decomposition import TruncatedSVD



n_comp = 30

svd = TruncatedSVD(n_components=n_comp, random_state=42)

svd_tfidf = svd.fit_transform(vz_sample)
svd_tfidf.shape
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
tsne_tfidf
output_notebook()

plot_tfidf = bp.figure(plot_width=700, plot_height=600,

                       title="tf-idf clustering of the item description",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)
combined_sample = combined_sample.reset_index(drop=True)
tfidf_df = pd.DataFrame(tsne_tfidf, columns=['x', 'y'])

tfidf_df['description'] = combined_sample['item_description']

tfidf_df['tokens'] = combined_sample['tokens']

tfidf_df['category'] = combined_sample['general_cat']
tfidf_df.head()
plot_tfidf.scatter(x='x', y='y', source=tfidf_df, alpha=0.7)

hover = plot_tfidf.select(dict(type=HoverTool))

hover.tooltips={"description": "@description", "tokens": "@tokens", "category":"@category"}

show(plot_tfidf)
from sklearn.cluster import MiniBatchKMeans



num_clusters = 30 # need to be selected wisely

kmeans_model = MiniBatchKMeans(n_clusters=num_clusters,

                               init='k-means++',

                               n_init=1,

                               init_size=1000, batch_size=1000, verbose=0, max_iter=1000)
kmeans = kmeans_model.fit(vz)

kmeans_clusters = kmeans.predict(vz)

kmeans_distances = kmeans.transform(vz)
# sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

# terms = vectorizer.get_feature_names()



# for i in range(num_clusters):

#     print("Cluster %d:" % i)

#     aux = ''

#     for j in sorted_centroids[i, :10]:

#         aux += terms[j] + ' | '

#     print(aux)

#     print() 
# repeat the same steps for the sample

kmeans = kmeans_model.fit(vz_sample)

kmeans_clusters = kmeans.predict(vz_sample)

kmeans_distances = kmeans.transform(vz_sample)

# reduce dimension to 2 using tsne

tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",

"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",

"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",

"#52697d", "#194196", "#d27c88", "#36422b", "#b68f79"])
#combined_sample.reset_index(drop=True, inplace=True)

kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])

kmeans_df['cluster'] = kmeans_clusters

kmeans_df['description'] = combined_sample['item_description']

kmeans_df['category'] = combined_sample['general_cat']

#kmeans_df['cluster']=kmeans_df.cluster.astype(str).astype('category')
plot_kmeans = bp.figure(plot_width=700, plot_height=600,

                        title="KMeans clustering of the description",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)
source = ColumnDataSource(data=dict(x=kmeans_df['x'], y=kmeans_df['y'],

                                    color=colormap[kmeans_clusters],

                                    description=kmeans_df['description'],

                                    category=kmeans_df['category'],

                                    cluster=kmeans_df['cluster']))



plot_kmeans.scatter(x='x', y='y', color='color', source=source)

hover = plot_kmeans.select(dict(type=HoverTool))

hover.tooltips={"description": "@description", "category": "@category", "cluster":"@cluster" }

show(plot_kmeans)
cvectorizer = CountVectorizer(min_df=4,

                              max_features=180000,

                              tokenizer=tokenize,

                              ngram_range=(1,2))
cvz = cvectorizer.fit_transform(combined_sample['item_description'])
cvz
lda_model = LatentDirichletAllocation(n_components=100,

                                      learning_method='online',

                                      max_iter=20,

                                      random_state=42)
X_topics = lda_model.fit_transform(cvz)
lda_model.components_.shape
sorting = np.argsort(lda_model.components_,axis=1)[::-1]

feature_names = np.array(cvectorizer.get_feature_names())
fig, ax = plt.subplots(1,2,figsize=[10,12])

topic_names = ["{:>2} ".format(i)+" ".join(words) for i,words in enumerate(feature_names[sorting[:,:2]])]



for col in [0,1]:

    start = col * 50

    end = (col+1) * 50

    ax[col].barh(np.arange(50),np.sum(X_topics,axis=0)[start:end])

    ax[col].set_yticks(np.arange(50))

    ax[col].set_yticklabels(topic_names[start:end],ha='left',va='top')

    ax[col].invert_yaxis()

    ax[col].set_xlim(0,1000)

    yax = ax[col].get_yaxis()

    yax.set_tick_params(pad=130)

plt.tight_layout()