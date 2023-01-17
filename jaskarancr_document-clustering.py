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
df=pd.read_csv('/kaggle/input/news-article/news_articles.csv')
df_clicks=pd.read_csv('/kaggle/input/click-data/clicks_new - clicks.csv')
df.head(5)
df.shape
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
titles=list(df['Title'])
content=list(df['Content'])
titles[:10]
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in content:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=10, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(content) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
from sklearn.cluster import KMeans
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(KMeans(n_clusters=k).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
find_optimal_clusters(tfidf_matrix, 10)

from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

clusters=km.fit_predict(tfidf_matrix)

clusters
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    
plot_tsne_pca(tfidf_matrix, clusters)
def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
            
get_top_keywords(tfidf_matrix, clusters, tfidf_vectorizer.get_feature_names(), 15)
from collections import Counter
Counter(clusters)
new_article=['KARACHI: Seven Pakistani soldiers were killed in two separate terror attacks in the restive Balochistan province, an official statement said on Tuesday. Terrorists targeted a vehicle of the Frontier Corps using improvised explosive devices in the Pir Ghaib area on Monday night, killing six Pakistan Army soldiers, said Inter-Services Public Relations (ISPR), the media wing of the Pakistani military. In a separate incident in Balochistan Kech, another soldier was killed during an exchange of fire with the militants.Resource-rich Balochistan in southwestern Pakistan borders Afghanistan and Iran, But it is also Pakistan largest and poorest province, rife with ethnic, sectarian and separatist insurgencies.']
unseen_tfidf = tfidf_vectorizer.transform(new_article)
km.predict(unseen_tfidf)
cluster0_index= []
cluster1_index= []
cluster2_index= []
cluster3_index= []
cluster4_index= []
for i in range(0,len(clusters)-1):
    if clusters[i]==0:
        cluster0_index.append(i)
for i in range(0,len(clusters)-1):
    if clusters[i]==1:
        cluster1_index.append(i)
for i in range(0,len(clusters)-1):
    if clusters[i]==2:
        cluster2_index.append(i)
for i in range(0,len(clusters)-1):
    if clusters[i]==3:
        cluster3_index.append(i)
for i in range(0,len(clusters)-1):
    if clusters[i]==4:
        cluster4_index.append(i)
len(cluster2_index)
#randomly
import random
sampling0 = random.choices(cluster0_index, k=2)
sampling1 = random.choices(cluster1_index, k=2)
sampling2 = random.choices(cluster2_index, k=2)
sampling3 = random.choices(cluster3_index, k=2)
sampling4 = random.choices(cluster4_index, k=2)
show_list=[sampling0,sampling1,sampling2,sampling3,sampling4]
recommend_index=[]
for x in show_list:
    for ele in x:
        recommend_index.append(ele)
n=0
print('Articles Recommended for you :) ')
for i in recommend_index:
    n=n+1
    print('Article ',n,': ',df['Title'][i])
#Breaking news
df1=df.copy()
df1.isna(). sum()
df1=df1.dropna(subset=['Date'])
df1.shape
df1[df1['Date'] == ' Saranya Ponvannan']
df1=df1.drop(544)
import datetime
df1['Date'] = df1['Date'].astype('datetime64[ns]') 
df1=df1.sort_values(by="Date")
df1.head(5)
list_article=list(df1['Article_Id'])
cluster0_date_index=[]
cluster1_date_index=[]
cluster2_date_index=[]
cluster3_date_index=[]
cluster4_date_index=[]
for i in cluster0_index:
    if i in list_article:
        cluster0_date_index.append(i)
for i in cluster1_index:
    if i in list_article:
        cluster1_date_index.append(i)
for i in cluster2_index:
    if i in list_article:
        cluster2_date_index.append(i)
for i in cluster3_index:
    if i in list_article:
        cluster3_date_index.append(i)
for i in cluster4_index:
    if i in list_article:
        cluster4_date_index.append(i)
import operator
dict0={}
dict1={}
dict2={}
dict3={}
dict4={}
for i in cluster0_date_index:
    dict0[i]=df1['Date'][i]
dict0 = sorted(dict0.items(), key=operator.itemgetter(1))
for i in cluster1_date_index:
    dict1[i]=df1['Date'][i]
dict1 = sorted(dict1.items(), key=operator.itemgetter(1))
for i in cluster2_date_index:
    dict2[i]=df1['Date'][i]
dict2 = sorted(dict2.items(), key=operator.itemgetter(1))
for i in cluster3_date_index:
    dict3[i]=df1['Date'][i]
dict3 = sorted(dict3.items(), key=operator.itemgetter(1))
for i in cluster4_date_index:
    dict4[i]=df1['Date'][i]
dict4 = sorted(dict4.items(), key=operator.itemgetter(1))
#breaking news recommendation from each cluster
x1=dict0[-2:][0][0]
x2=dict0[-2:][1][0]
x3=dict1[-2:][0][0]
x4=dict1[-2:][1][0]
x5=dict2[-2:][0][0]
x6=dict2[-2:][1][0]
x7=dict3[-2:][0][0]
x8=dict3[-2:][1][0]
x9=dict4[-2:][0][0]
x10=dict4[-2:][1][0]
list_breaking=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
n=0
print("Breaking News Recommendation:):")
for i in list_breaking:
    n=n+1
    print('Article ',n,': ',df['Title'][i])

df['click_count']=list(df_clicks['Total_clicks'])
df.head(5)
import operator
dict_click0={}
dict_click1={}
dict_click2={}
dict_click3={}
dict_click4={}
for i in cluster0_index:
    dict_click0[i]=df['click_count'][i]
dict_click0 = sorted(dict_click0.items(), key=operator.itemgetter(1))
for i in cluster1_index:
    dict_click1[i]=df['click_count'][i]
dict_click1 = sorted(dict_click1.items(), key=operator.itemgetter(1))
for i in cluster2_index:
    dict_click2[i]=df['click_count'][i]
dict_click2 = sorted(dict_click2.items(), key=operator.itemgetter(1))
for i in cluster3_index:
    dict_click3[i]=df['click_count'][i]
dict_click3 = sorted(dict_click3.items(), key=operator.itemgetter(1))
for i in cluster4_index:
    dict_click4[i]=df['click_count'][i]
dict_click4 = sorted(dict_click4.items(), key=operator.itemgetter(1))
dict_click4
#breaking news recommendation from each cluster
y1=dict_click0[-2:][0][0]
y2=dict_click0[-2:][1][0]
y3=dict_click1[-2:][0][0]
y4=dict_click1[-2:][1][0]
y5=dict_click2[-2:][0][0]
y6=dict_click2[-2:][1][0]
y7=dict_click3[-2:][0][0]
y8=dict_click3[-2:][1][0]
y9=dict_click4[-2:][0][0]
y10=dict_click4[-2:][1][0]
list_famous=[y1,y2,y3,y4,y5,y6,y7,y8,y9,y10]
n=0
print("Frequently viewed News Articles:")
for i in list_famous:
    n=n+1
    print('Article ',n,': ',df['Title'][i])





import pickle
objects = []
with (open("/kaggle/input/embed-mean/emeddings_bert_mean.txt", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
embed_list=[]
for x in objects[0]:
    emb_np=x.cpu().detach().numpy()
    embed_list.append(emb_np)
len(embed_list)
from sklearn.cluster import KMeans
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(KMeans(n_clusters=k).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
find_optimal_clusters(embed_list, 10)
from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

clusters=km.fit_predict(embed_list)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
emd_arr=np.array(embed_list)
def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:])
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:]))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    
plot_tsne_pca(emd_arr, clusters)
from collections import Counter
Counter(clusters)
cluster0_index= []
for i in range(0,len(clusters)-1):
    if clusters[i]==0:
        cluster0_index.append(i)

cluster0_articles=[]
for i in cluster0_index:
    cluster0_articles.append(df['Content'][i])
len(cluster0_articles)
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=10, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(cluster0_articles) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
def display_scores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))
display_scores(tfidf_vectorizer, tfidf_matrix)
cluster1_index= []
for i in range(0,len(clusters)-1):
    if clusters[i]==1:
        cluster1_index.append(i)
cluster1_articles=[]
for i in cluster1_index:
    cluster1_articles.append(df['Content'][i])
len(cluster1_articles)
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=10, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(cluster1_articles) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
display_scores(tfidf_vectorizer, tfidf_matrix)
cluster2_index= []
for i in range(0,len(clusters)-1):
    if clusters[i]==2:
        cluster2_index.append(i)
cluster2_articles=[]
for i in cluster2_index:
    cluster2_articles.append(df['Content'][i])
len(cluster2_articles)
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=5, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(cluster2_articles) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
display_scores(tfidf_vectorizer, tfidf_matrix)
cluster3_index= []
for i in range(0,len(clusters)-1):
    if clusters[i]==3:
        cluster3_index.append(i)
cluster3_articles=[]
for i in cluster3_index:
    cluster3_articles.append(df['Content'][i])
len(cluster3_articles)
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=5, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(cluster3_articles) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
display_scores(tfidf_vectorizer, tfidf_matrix)
cluster4_index= []
for i in range(0,len(clusters)-1):
    if clusters[i]==4:
        cluster4_index.append(i)
cluster4_articles=[]
for i in cluster4_index:
    cluster4_articles.append(df['Content'][i])
len(cluster4_articles)
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=5, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(cluster4_articles) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
display_scores(tfidf_vectorizer, tfidf_matrix)
