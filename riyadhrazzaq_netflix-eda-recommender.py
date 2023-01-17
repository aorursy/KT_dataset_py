import numpy as np
import pandas as pd
import subprocess as sp
import matplotlib.pyplot as plt
from collections import Counter
import wordcloud
# loading data
df = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
n_samples,n_feats = df.shape
df
df.columns

print(df.shape[0])
df.groupby(['type']).count()['show_id']

df.isnull().sum()
# movie and series distribution
dist = df.groupby(df['type']).count()['show_id'].reset_index()
fig, ax = plt.subplots(facecolor='white',figsize=(10,10))
ax.bar(dist['type'],dist['show_id'])
# dist of movie or series running in countries
countries = []
for i in df['country']:
    if isinstance(i,str):
        l = i.split(',')
        for i in range(len(l)):
            l[i] = l[i].strip()
        countries = countries + l
dist = dict(Counter(countries))
dist20 = {k:v for k,v in sorted(dist.items(),key=lambda x: x[1],reverse=True)[:20]}
fig, ax = plt.subplots(facecolor='white',figsize=(10,20))
ax.barh(list(dist20.keys()),dist20.values())
ax.grid(True,axis='x')
ax.set_xlabel("Number of shows running")
plt.title("Top 20 Country by Show Available")
plt.show()
# release year distribution, grouped by 10 years
years= df['release_year'].values
maxyrs = np.max(years)
minyrs = np.min(years)
dist = {k:0 for k in np.arange(1920,2021,10)}
for i in years:
    dist[int(str(int(i / 10))+'0')] += 1
xticklabels = [str(i)+'s' for i in dist.keys()]
fig, ax = plt.subplots(facecolor='white', figsize= (20,10))
ax.bar(dist.keys(),dist.values())
ax.set_xlabel('Release Years')
ax.set_ylabel('Running Shows')
ax.set_xticks(list(dist.keys()))
ax.set_xticklabels(xticklabels)
plt.show()

# rating distribution
dist = df.groupby(['rating']).count().reset_index()[['rating','show_id']]
fig, ax = plt.subplots(facecolor='white',figsize=(10,8))
ax.bar(dist['rating'],dist['show_id'])
ticks = np.arange(len(dist['rating']))
ax.set_xticks(ticks)
ax.set_xticklabels(dist.rating)
ax.set_xlabel('Content Ratings')
ax.set_ylabel('Shows Count')
plt.show()
# tags dist
tmp_tags = []
for i in df['listed_in']:
    tags = [c.strip() for c in i.split(',')]
    tmp_tags = tmp_tags + tags

dist = {k:v for k,v in sorted(dict(Counter(tmp_tags)).items(),key=lambda x: x[1])}

fig, ax = plt.subplots(facecolor='white',figsize=(10,20))
yticks = sorted(np.arange(len(dist.keys())))
ax.barh(list(dist.keys()),dist.values())
ax.set_yticks(yticks)
ax.set_yticklabels(dist.keys())
ax.set_frame_on(False)
ax.grid(True,axis='x')

plt.title('Tags Count')
plt.show()
# description
wc = wordcloud.WordCloud(width=2000,height=2000)
texts = " ".join([w for w in df.description])
wc.generate(texts)
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(wc, interpolation='bilinear')
ax.axis("off")
plt.show()
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
es = EnglishStemmer(ignore_stopwords=False)
sw = set(stopwords.words('english'))
def clean(sent):
#     cleaned = [es.stem(w) for w in sent.split() if w not in sw]
    sent = sent.lower()
    cleaned = [w for w in sent.split() if w not in sw]
    return " ".join(cleaned)

for i in range(df.shape[0]):
    df.loc[i,'description'] = clean(df.loc[i,'description'])
try:
    import fasttext.util
except:
    !pip install fasttext
    import fasttext.util

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from scipy import spatial
!pip install python-levenshtein
from Levenshtein import ratio
import random
from sklearn.metrics.pairwise import cosine_similarity

fasttext.util.download_model('en',if_exists='ignore') # english
ft = fasttext.load_model('cc.en.300.bin')
s1 = ft.get_sentence_vector('I am sad').reshape(1,-1)
s2 = ft.get_sentence_vector('I am happy').reshape(1,-1)
print(cosine_similarity(s1,s2)[0][0])
ITEM = random.randint(0,df.shape[0])
# ITEM = 5285
s1 = ft.get_sentence_vector(df.loc[ITEM,'description']).reshape(1,-1)
print('==================Target==================\nTitle: %s\nCategory: %s\nDescription: %s\n===================================='%(df.loc[ITEM,'title'],df.loc[ITEM,'listed_in'],df.loc[ITEM,'description']))
sims = []
for i in np.random.randint(0,df.shape[0]-1,100):
    s2 = ft.get_sentence_vector(df.loc[i,'description']).reshape(1,-1)
    sim = cosine_similarity(s1,s2)
    sims.append((i,sim[0][0]))
top = sorted(sims,key=lambda x: x[1],reverse=True)[:5]
for tpl in top[:10]:
    print('==================%.2f==================\nTitle: %s\nCategory: %s\nDescription: %s\n===================================='%(tpl[1],df.loc[tpl[0],'title'],df.loc[tpl[0],'listed_in'],df.loc[tpl[0],'description']))
tfvec = TfidfVectorizer(ngram_range=(1,1))
tfvec.fit(df['description'])
# this needs to be changed for tfvec
ITEM = random.randint(0,df.shape[0])
# ITEM = 5285
s1 = ft.get_sentence_vector(df.loc[ITEM,'description']).reshape(1,-1)
print('==================Target==================\nTitle: %s\nCategory: %s\nDescription: %s\n===================================='%(df.loc[ITEM,'title'],df.loc[ITEM,'listed_in'],df.loc[ITEM,'description']))
sims = []
for i in np.random.randint(0,df.shape[0]-1,100):
    s2 = ft.get_sentence_vector(df.loc[i,'description']).reshape(1,-1)
    sim = cosine_similarity(s1,s2)
    sims.append((i,sim[0][0]))
top = sorted(sims,key=lambda x: x[1],reverse=True)[:5]
for tpl in top[:10]:
    print('==================%.2f==================\nTitle: %s\nCategory: %s\nDescription: %s\n===================================='%(tpl[1],df.loc[tpl[0],'title'],df.loc[tpl[0],'listed_in'],df.loc[tpl[0],'description']))
df = df.dropna(subset=['rating'],axis=0).reset_index()
df.drop(columns=['index'],inplace=True)
from sklearn.preprocessing import OneHotEncoder
def customOneHotEncoder(X):
    """
    Custom one hot encoding of 'listed_in' features
    Parameters
    ----------
    X: dataframe like 2d array
    
    Returns
    -------
    X: Dataframe
    """
    Nsamples = X.shape[0]
    tmp_tags = []
    for i in X['listed_in']:
        tags = [c.strip() for c in i.split(',')]
        tmp_tags = tmp_tags + tags
    unq_tags = ['listed_in_'+t for t in set(tmp_tags)]
    df = pd.DataFrame(data=np.zeros((Nsamples,len(unq_tags))),columns=unq_tags)
    
    for i in range(Nsamples):
        tags = [c.strip() for c in X.loc[i,'listed_in'].split(',')]
        for t in tags:
            df.loc[i,'listed_in_'+t] = 1
    return pd.DataFrame(df)
X = df[['type','listed_in','rating']]
X = customOneHotEncoder(X)
import sklearn.cluster as skc
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation

# KMeans
km = KMeans(n_clusters=30)
km.fit(X)
X['cluster'] = km.labels_
df['cluster'] = km.labels_
def predict(target, scope, n=5):
    """
    Predicts similar N movies by Cosine similarity with description. If target has attribute **cluster**, function will only match movies in the same cluster.
    Params
    ------
    target: Series. shape: (n_features,)
    scope: dataframe on which to search (n_samples,n_features)
    Returns
    -------
    Y: dataframe. columns: [rank, similarity] + X.columns
    """
    # tfvec = TfidfVectorizer(ngram_range=(1,3))
    
#     tf_target = tfvec.transform([target['description']])
    cluster = True if 'cluster' in scope.columns else False
    sims = []

    if cluster:

        idx = scope[scope['cluster']==target['cluster']].index

        # X = tfvec.fit_transform(scope.loc[idx,'description'])
        # tf_target = tfvec.transform([target['description']])
        ft_target = ft.get_sentence_vector(target['description']).reshape(1,-1)

        for i in range(len(idx)):
            if scope.loc[idx[i],'show_id'] == target['show_id']:
                continue
            # tf_text = X[i]
            # tf_sim = cosine_similarity(tf_text,tf_target)
            # sims.append((scope.loc[idx[i],'show_id'],tf_sim,scope.loc[idx[i],'cluster']))
            # print(idx[i])
            
            ft_text = ft.get_sentence_vector(scope.loc[idx[i],'description']).reshape(1,-1)
            ft_sim = cosine_similarity(ft_target,ft_text)
            sims.append((scope.loc[idx[i],'show_id'],ft_sim,scope.loc[idx[i],'cluster']))
            
    else:
        tfvec.fit_transform(scope.loc[idx,'description'])
        tf_target = tfvec.transform([target['description']])
        
        for i in range(scope.shape[0]):
            text = scope.loc[i,'description']
            tf_text = tfvec.transform([text])
            tf_sim = cosine_similarity(tf_text,tf_target)
            sims.append((scope.loc[idx[i],'show_id'],tf_sim,scope.loc[idx[i],'cluster']))
            
    top = sorted(sims,key=lambda x: x[1],reverse=True)
    return top[:5]

def showbyid(df,obj):
    """
    Returns movies in a dataframe.
    """
    idx = [i[0] for i in obj]
    confidence = [i[1][0][0] for i in obj]
    
    tmp = pd.DataFrame(columns=df.columns)
    for i in idx:
        tmp = tmp.append(df.loc[df['show_id']==i,:])
    tmp['confidence'] = confidence
    return tmp.reset_index()
test = random.randint(0,df.shape[0]-1)
print("-----Query--\n",df.loc[test])
res = predict(df.loc[test],df)
similar_movies = showbyid(df,res)
similar_movies