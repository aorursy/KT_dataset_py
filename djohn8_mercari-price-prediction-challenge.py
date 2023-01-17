import numpy as np
import pandas as pd
import time
import os

import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly_express as px
import plotly.graph_objects as go

import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Train data
train = pd.read_csv('/kaggle/input/mercari-price-prediction-train-and-test-data/train.tsv', sep='\t')
test = pd.read_csv('/kaggle/input/mercari-price-prediction-train-and-test-data/test.tsv', sep='\t')

display(train.head())
print('Train data:', train.shape)
#display(train.shape)

print('Test data shape: ', test.shape)
train.columns
train.info()
train.isnull().sum()[train.isnull().sum().values > 0]
train.price.describe()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

ax1.hist(train['price'], range=[0,250], bins=50, edgecolor='w')
ax1.set_xlabel('price')
ax1.set_ylabel('distribution')
ax1.set_title('histogram of price')

ax2.hist(np.log1p(train['price']), range=[0,10], bins=30, edgecolor='w')
ax2.set_xlabel('Log of price')
ax2.set_ylabel('distribution')
ax2.set_title('histogram of Log price')
plt.show()
train['logprice'] = np.log1p(train['price'])
train.head(5)
train['shipping'].value_counts(normalize=True)
plt.figure(figsize=(15,5))
train.groupby('shipping')['price'].plot(kind='hist', range=[0,100], bins=25, alpha=0.7)
plt.legend(labels=['buyer', 'seller'])
plt.xlabel('price')
plt.ylabel('distribution')
plt.title('variations across buyer and seller')
plt.show()
print('Number of unique categories is: {}\n'.format(len(train['category_name'].value_counts())))
print('Top 10 categories: \n{}'.format(train['category_name'].value_counts()[:10]))
# split the categories

# text = 'Women/Athletic Apparel/Pants, Tights, Leggings'
# text.split('/')

def split_categories(text):
    try:
        return text.split('/')
    # if no category name
    except: 
        return ['no label', 'no label', 'no label']
train['general_cat'] = train['category_name'].apply(lambda x: split_categories(x)[0])
train['sub_cat1'] = train['category_name'].apply(lambda x: split_categories(x)[1])
train['sub_cat2'] = train['category_name'].apply(lambda x: split_categories(x)[2])

train.head(10)
plt.figure(figsize=(15,5))
plt.bar(x = train['general_cat'].value_counts().index.values, height = train['general_cat'].value_counts().values)
plt.xlabel('General categories')
plt.ylabel('Count')
plt.show()
# show in plotly

trace = go.Bar(x = train['general_cat'].value_counts().index.values, 
               y = train['general_cat'].value_counts().values,
                text = round(train['general_cat'].value_counts(normalize=True)*100,2)
              )
layout = go.Layout(dict(
                        title = 'Number of items by general category'),
                        xaxis= dict(title = 'general categories'),
                        yaxis= dict(title = 'Count')
                  )
fig  = go.Figure(data =trace, layout = layout)
py.iplot(fig)
# show in plotly

trace = go.Bar(x = train['sub_cat1'].value_counts().index.values[:20], 
               y = train['sub_cat1'].value_counts().values[:20],
                text = round(train['sub_cat1'].value_counts(normalize=True)*100,2)
              )
layout = go.Layout(dict(
                        title = 'Number of items by sub_category 1'),
                        xaxis= dict(title = 'sub_category 1'),
                        yaxis= dict(title = 'Count')
                  )
fig  = go.Figure(data =trace, layout = layout)
py.iplot(fig)
# show in plotly

trace = go.Bar(x = train['sub_cat2'].value_counts().index.values[:20], 
               y = train['sub_cat2'].value_counts().values[:20],
                text = round(train['sub_cat2'].value_counts(normalize=True)*100,2),
                             )
layout = go.Layout(dict(
                        title = 'Number of items by sub_category 2'),
                        xaxis= dict(title = 'sub_category 2'),
                        yaxis= dict(title = 'Count'),
                        )
fig  = go.Figure(data =trace, layout = layout)
py.iplot(fig)
gen_cat = train['general_cat'].unique()
x = [train.loc[train['general_cat'] == cat, 'price'] for cat in gen_cat]

trace = [go.Box(x = np.log1p(x[i]), name = gen_cat[i]) for i in range(len(gen_cat))]
layout = dict(
            title = 'Price distrbution across general category',
            xaxis = dict(title='distribution'),
            yaxis = dict(title = 'category')
            )
fig = go.Figure(data = trace, layout=layout)
py.iplot(fig)
train.isnull().sum()[train.isnull().sum().values > 0]
# fill the columns that have missing values

def handle_missing_values(df):
    df['category_name'].fillna(value = 'missing', inplace=True)
    df['brand_name'].fillna(value = 'missing', inplace=True)
    df['item_description'].replace('No description yet', 'missing', inplace=True)
    df['item_description'].fillna(value = 'missing', inplace=True)
print('Number of unique brands is : ', len(train['brand_name'].value_counts()))
print('Number of unique brands with count > 1 : ', len(train['brand_name'].value_counts()[train['brand_name'].value_counts() > 1]))
train['item_description']
text = train.iloc[1482534]['item_description']
text
from nltk.corpus import stopwords
stop = stopwords.words('english')
import re

def tokenize_text(text):
    regex = re.compile(r'[a-zA-Z]{3,}')
    txt = regex.findall(str(text).lower())
    #tokens = [t for t in txt if t not in stop and len(t)>3]
    tokens = [t for t in txt if t not in stop and len(t)>3]
    return tokens
# from sklearn.feature_extraction.text import CountVectorizer
# #cv = CountVectorizer(stop_words='english', token_pattern= r'\b[^\d\W]+\b')
# cv = CountVectorizer(stop_words='english', token_pattern= r'[a-zA-Z]{3,}')
# #cv = CountVectorizer()

# def word_count(text):
#     try:
#         cv.fit([text]) ## if this doesnt work, try 'text' instead of '[text]'
#         return len(cv.get_feature_names())
#     except:
#         return 0
#cv = CountVectorizer(stop_words='english', token_pattern= r'\b[^\d\W]+\b')
# cv = CountVectorizer(stop_words='english', token_pattern= r'[a-zA-Z]{3,}')
# #cv = CountVectorizer(stop_words='english')

# def word_token(text):
#     try:
#         cv.fit([text]) ## if this doesnt work, try 'text' instead of '[text]'
#         tokens = cv.get_feature_names()
#         return tokens
#     except:
#         return 0
# trace = go.Scatter(x = df['desc_length'], y = df['price'],
#                   mode = 'lines+markers')
# layout = go.Layout(dict(title='price vz description'))

# fig = go.Figure(data=[trace], layout=layout)
# fig.show()
train_sample = train.sample(frac=0.001, random_state=123)
train_sample['tokens'] = train_sample['item_description'].apply(lambda x: tokenize_text(x))
train_sample['desc_length'] = train_sample['tokens'].apply(lambda x: len(x))

train_sample.head(10)

for description, tokens, length in zip(train_sample['item_description'], train_sample['tokens'], train_sample['desc_length']):
    print('Item description: ', description)
    print('Word tokens', tokens)
    print('Description length: ', length)
    print('\n')
# train['tokens'] = train['item_description'].apply(lambda x: tokenize_text(x))
# train['desc_length'] = train['tokens'].apply(lambda x: len(x))

# train.head(10)

# for description, tokens, length in zip(train_sample['item_description'], train_sample['tokens'], train_sample['desc_length']):
#     print('Item description: ', description)
#     print('Word tokens', tokens)
#     print('Description length: ', length)
#     print('\n')
# train_sample = train.sample(frac=0.001, random_state=10)
# train_sample['tokens'] = train_sample['item_description'].apply(lambda x: tokenize_text([x]))
# train_sample['desc_length'] = train_sample['item_description'].apply(lambda x: word_count([x]))
# train_sample[['tokens','desc_length']].head(10)
import datetime
import time

start = time.time()
print('Start time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
train['tokens'] = train['item_description'].apply(lambda x: tokenize_text(x))
train['desc_length'] = train['tokens'].apply(lambda x: len(x))

print('time taken for train:', time.time() - start)
print('End time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

# for test
start = time.time()
print('Start time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
test['tokens'] = test['item_description'].apply(lambda x: tokenize_text(x))
test['desc_length'] = test['tokens'].apply(lambda x: len(x))
print('time taken for test:', time.time() - start)
# print('End time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

train.head(10)
train.to_csv('train_desc.csv',index=False)
test.to_csv('test_desc.csv',index=False)
# train = pd.read_csv('/kaggle/input/mercari-price-suggestion-with-desc/train_desc.csv')
# test = pd.read_csv('/kaggle/input/mercari-price-suggestion-with-desc/test_desc.csv')
# train.head()
df = train.groupby('desc_length')['price'].mean().reset_index()
df
trace = go.Scatter(
                    x = df['desc_length'],
                    y = df['price'],
                    mode = 'lines+markers',
                    )

layout = go.Layout(dict(title = 'Average price per description length',
                       xaxis = dict(title = 'Description length'),
                       yaxis = dict(title = 'price')
                       )
                  )
fig = go.Figure(data = [trace], layout=layout)
py.iplot(fig)
display(train.isnull().sum())
display(train.shape)
# removing the entries which do not have a 'item description'

train_new = train[pd.notnull(train['item_description'])]
display(train_new.isnull().sum())
display(train_new.shape)
train['general_cat'].value_counts()
from wordcloud import WordCloud
from collections import Counter

cat_desc = dict()
for cat in train_sample['general_cat'].value_counts().index.tolist():
    text = " ".join(train_sample.loc[train_sample['general_cat'] == cat, 'item_description'].values)
    cat_desc[cat]= tokenize_text(text)

womens100 = Counter(cat_desc['Women']).most_common(100)
beauty100 = Counter(cat_desc['Beauty']).most_common(100)
kids100 = Counter(cat_desc['Kids']).most_common(100)
electronics100 = Counter(cat_desc['Electronics']).most_common(100)
keys = [k for (k,v) in womens100]
text = " ".join(keys)
text
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#optional - to remove the word 'shipping' from the wordcloud
stopwords = set(STOPWORDS)
#print(stopwords)
stopwords.update(['shipping', 'Shipping', "shipping'"])
#optional

def create_wordcloud(text):
    wordcloud = WordCloud(background_color='white', stopwords=stopwords,
                          max_font_size=30).generate(text)
    return wordcloud
fig, ax = plt.subplots(2,2, figsize=(15, 10))

# wordcloud requires input text to be a string

ax[0,0].imshow(create_wordcloud(str(womens100)), interpolation='bilinear')
ax[0,0].set_title('women', fontsize=25)

ax[0,1].imshow(create_wordcloud(str(beauty100)), interpolation='bilinear')
ax[0,1].set_title('beauty', fontsize=25)

ax[1,0].imshow(create_wordcloud(str(kids100)), interpolation='bilinear')
ax[1,0].set_title('kids', fontsize=25)

ax[1,1].imshow(create_wordcloud(str(electronics100)), interpolation='bilinear')
ax[1,1].set_title('Electronics', fontsize=25)

plt.tight_layout(pad =1)
plt.show()
# Some random tests

import operator
temp = Counter(cat_desc['Women'])
#sorted(temp.items(), key=operator.itemgetter(1), reverse=True)

lofwords=[]
for k, v  in temp.items():
    if v > 10:
        lofwords.append(k)

len(lofwords)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10,\
                             max_features= 180000, \
                             tokenizer = tokenize_text, \
                             ngram_range=(1,2)
                            )
train_new.isnull().sum()
# Apply to train and test together
# use train_new, which has no item description missing

start = time.time()
all_desc = np.append(train_new['item_description'].values, test['item_description'].values)
vz = vectorizer.fit_transform(all_desc)
print('Time taken:', time.time() - start)
# saving the model file
import pickle
pickle.dump(vz, open('tfidf_vectorizer_combined_whole.pkl', 'wb'))
# extract features
# zip the features and the correspondinf IDF scores together
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

#create dataframe
tfidf = pd.DataFrame(tfidf.items(), columns=['features', 'idf_score'])
tfidf.sort_values('idf_score', ascending=False).head(10)
print('Shape of the vectorized implementation of item description: ',vz.shape)
print('Dimension of the Tf-Idf dataframe is :', tfidf.shape)
train_copy = train_new.copy()
test_copy = test.copy()

train_copy['is_train'] = 1
test_copy['is_train'] = 0


combined_df = pd.concat([train_copy, test_copy], sort=False)

sample_size= 15000
combined_sample = combined_df.sample(n=sample_size, random_state=1)

# applying tfidf vectorizer
#vz_sample = vectorizer.fit_transform(combined_sample['item_description'].values)
vz_sample = vectorizer.transform(combined_sample['item_description'].values)

print('Shape of the sample tfidf matrix is: ', vz_sample.shape)
from sklearn.decomposition import TruncatedSVD

n_comp= 30
svd = TruncatedSVD(n_components=n_comp, random_state=1)
svd_tfidf = svd.fit_transform(vz_sample)
print('Shape of svd_tfidf matrix is:', svd_tfidf.shape)
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
print('Shape of tsne_tfidf matrix is:', tsne_tfidf.shape)
# saving the tsne model
pickle.dump(tsne_tfidf, open('tsne_svd_tfidf_combined_sample.pkl', 'wb'))
combined_sample.head()
combined_sample.reset_index(drop=True, inplace=True)
combined_sample.head()
# After SVD and T-SNE on a sample of the combined train and test datasets

tfidf_df = pd.DataFrame(tsne_tfidf, columns=['pc1', 'pc2'])
tfidf_df['item_description'] = combined_sample['item_description']
tfidf_df['tokens'] = combined_sample['tokens']
tfidf_df['category'] = combined_sample['general_cat']

tfidf_df.head()
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600,
                       title="tf-idf clustering of the item description",
    tools="pan,wheel_zoom,box_zoom,reset,hover, save",
    x_axis_type=None, y_axis_type=None, min_border=1)
plot_tfidf.scatter(x='pc1', y='pc2', source=tfidf_df, alpha=0.7)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"description": "@item_description", "tokens": "@tokens", "category":"@category"}
show(plot_tfidf)
#plt.scatter(tfidf_df['pc1'], tfidf_df['pc2'])
from sklearn.cluster import MiniBatchKMeans

num_clusters=30

kmeans_model = MiniBatchKMeans(n_clusters = num_clusters, 
                               init='k-means++',
                               n_init=1,
                               init_size=1000, 
                               batch_size=1000, 
                               max_iter=1000,
                               verbose=1
                              )
# Using the original tfidf vectorizer which will be fit on the combined "whole" train and test
vectorizer = TfidfVectorizer(min_df=10,\
                             max_features= 180000, \
                             tokenizer = tokenize_text, \
                             ngram_range=(1,2)
                            )

all_desc = np.append(train_new['item_description'].values, test['item_description'].values)
vz = vectorizer.fit_transform(all_desc)
print('Number of features or tokens ', len(vectorizer.get_feature_names()))
# Applying the kmeans clustering to the tfidf matrix, to reduce dimensions
kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans_model.predict(vz)
kmeans_distances = kmeans_model.transform(vz)
# saving the model
pickle.dump(kmeans_model, open('kmeans_tfidf_combined_whole.pkl', 'wb'))
np.unique(kmeans_clusters, return_counts=True)
print(kmeans.cluster_centers_.shape)
terms = vectorizer.get_feature_names()
len(terms)
sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
sorted_centroids
for i in range(num_clusters):
    print('Cluster centroid: %d' %i)
    aux = ''
    #finding the top 10 words for every cluster
    for j in sorted_centroids[i, :10]:
        aux += terms[j] + ' | '
        
    print(aux)
print('Number of clusters:', np.unique(kmeans_clusters))
# This is the same as
print(np.unique(kmeans.labels_))
print('Shape of the vz_sample tfidf matrix', vz_sample.shape)

# Apply the Batched K-Means on the sample
kmeans = kmeans_model.fit(vz_sample)
# get the cluster centroids and mapping of features to the clusters
kmeans_clusters = kmeans.predict(vz_sample)
kmeans_distances = kmeans.transform(vz_sample)

# apply the T-SNE model to reduce dimension to 2, so it becomes easier to visualize
tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
# saving the model
pickle.dump(kmeans_model, open('tsne_kmeans_combined_sample.pkl', 'wb'))
# dimension has been shrunk to 2
tsne_kmeans.shape
#creata dataframe for the sample
kmeans_df = pd.DataFrame(tsne_kmeans, columns=['pc1', 'pc2'] )
kmeans_df['cluster']= kmeans_clusters
kmeans_df['description']= combined_sample['item_description']
kmeans_df['category']= combined_sample['general_cat']

kmeans_df.head()
plot_kmeans = bp.figure(plot_width=700, plot_height=600,
                        title="KMeans clustering of the description",
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    x_axis_type=None, y_axis_type=None, min_border=1)
colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
"#52697d", "#194196", "#d27c88", "#36422b", "#b68f79"])
source = ColumnDataSource(data=dict(x=kmeans_df['pc1'], y=kmeans_df['pc2'],
                                    color=colormap[kmeans_clusters],
                                    description=kmeans_df['description'],
                                    category=kmeans_df['category'],
                                    cluster=kmeans_df['cluster']))

plot_kmeans.scatter(x='x', y='y', color='color', source=source)
hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips={"description": "@description", "category": "@category", "cluster":"@cluster" }
show(plot_kmeans)
kmeans_df
# using seaborn
plt.figure(figsize=(20, 10))
sns.scatterplot(x=kmeans_df['pc1'], y=kmeans_df['pc2'], hue=kmeans_df['cluster'])
from sklearn.feature_extraction.text import CountVectorizer

cvectorizer = CountVectorizer(min_df=5, 
                             max_features=180000,
                             tokenizer = tokenize_text,
                             ngram_range = (1,2))
cvz = cvectorizer.fit_transform(combined_sample['item_description'])
# saving the model
pickle.dump(cvz, open('countVectorizer_for_lda.pkl', 'wb'))
import logging
logging.getLogger("lda").setLevel(logging.WARNING)

from sklearn.decomposition import LatentDirichletAllocation

num_topics=20
lda_model = LatentDirichletAllocation(n_components = num_topics, 
                                     max_iter=20, 
                                     random_state= 42, 
                                     learning_method='online'
                                     )
# input is a bag of words
X_topics = lda_model.fit_transform(cvz)
# saving the model
pickle.dump(lda_model, open('lda_model_countVectorizer.pkl', 'wb'))
print('Shape of the X_topics is:', X_topics.shape)
print(X_topics)
# getting the topic components
topic_word = lda_model.components_
print('Shape of topic_word is: ', topic_word.shape)
print(topic_word)
print(np.argsort(topic_word[0]))
print(np.argsort(topic_word[0])[:-10:-1])
np.array(vocabulary)[np.argsort(topic_word[0])[:-10:-1]]
vocabulary[1345]
topic_word = lda_model.components_ # these are the 20 topics words
vocabulary = cvectorizer.get_feature_names() # bag of words

topic_summaries=[]

n_top_words =10 # 10 most relevant words related to the topic

for i, topic_dist in enumerate(topic_word):
    indexes = np.argsort(topic_dist)[:-(n_top_words+1): -1] # getting the indexes of the top 10 words, in descending order
    topic_words = np.array(vocabulary)[indexes]
    print('topic %d' %i)
    print(' | '.join(topic_words))

# reduce dimensions using T-SNE
print('Dimension of X_topics is :', X_topics.shape)
tsne_lda = tsne_model.fit_transform(X_topics)
# saving the model
pickle.dump(tsne_lda, open('tsne_lda_model_combined_sample.pkl', 'wb'))
# convert to matrix to normalize the values across columns, such that sum across rows =1
unnormalized = np.matrix(X_topics)

doc_top_normalized = unnormalized / unnormalized.sum(axis=1)
doc_top_normalized
# check if normalized across rows
doc_top_normalized[0].sum()
# finding the most relevant topic for each item description
lda_keys=[]
for i, description in enumerate(combined_sample['item_description']):
    lda_keys += [doc_top_normalized[i].argmax()] # get the indexes of the topic with the highest probablisitic values
    
print(lda_keys[:10])

#create a Dataframe
lda_df = pd.DataFrame(tsne_lda, columns=['pc1', 'pc2'])
lda_df['description'] = combined_sample['item_description']
lda_df['category'] = combined_sample['general_cat']
lda_df['topic'] = lda_keys
lda_df['len_docs'] = combined_sample['tokens'].map(len)


lda_df
plot_lda = bp.figure(plot_width=700,
                     plot_height=600,
                     title="LDA topic visualization",
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    x_axis_type=None, y_axis_type=None, min_border=1)
source = ColumnDataSource(data=dict(x=lda_df['pc1'], y=lda_df['pc2'],
                                    color=colormap[lda_keys],
                                    description=lda_df['description'],
                                    topic=lda_df['topic'],
                                    category=lda_df['category']))

plot_lda.scatter(source=source, x='x', y='y', color='color')

hover = plot_kmeans.select(dict(type=HoverTool))
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips={"description":"@description",
                "topic":"@topic", "category":"@category"}
show(plot_lda)
def prepareLDAData():
    data = {
        'vocab': vocabulary,   # vocabulary = cvectorizer.get_feature_names()
        'doc_topic_dists': doc_top_normalized,
        'doc_lengths': list(lda_df['len_docs']),
        'term_frequency':cvectorizer.vocabulary_,
        'topic_term_dists': lda_model.components_
    } 
    return data
print('lda_model.components_ shape:', lda_model.components_.shape)
# get the count frequency of words in the entire corpus of item description in 'combined_sample'
cvectorizer.vocabulary_
import pyLDAvis

ldadata = prepareLDAData()
pyLDAvis.enable_notebook(sort=True)
prepared_data = pyLDAvis.prepare(**ldadata)
# saving the model
import pickle
with open('pyldavis_model.pkl', 'wb') as f:
    pickle.dump(prepared_data, f)
pyLDAvis.save_html(prepared_data, 'prepared_data_lda.html')