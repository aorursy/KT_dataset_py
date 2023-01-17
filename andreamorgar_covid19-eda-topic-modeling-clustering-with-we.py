# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd 
import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
import json
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import string  
import spacy
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from spacy.lang.en import English

import sklearn.metrics as metrics
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.cluster import AgglomerativeClustering, KMeans
import importlib


with open('/kaggle/input/CORD-19-research-challenge/metadata.readme', 'r') as f:
    data = f.read()
    print(data)
dirs = ['/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/',
        '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/',
        '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/',
        '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/']

filenames=[]
docs =[]
for d in dirs:
    for file in os.listdir(d):
        filename = d +file
        j = json.load(open(filename, 'rb'))
        
        paper_id =j['paper_id']
        #date =j['date']
        title = j['metadata']['title']
        authors = j['metadata']['authors']
        list_authors =[]
        for author in authors:
            if(len(author['middle'])==0):
                middle =""
            else :
                middle = author['middle'][0]
            _authors =author['first']+ " "+ middle +" "+ author['last']
            list_authors.append(_authors)
            
        try :
            abstract =  j['abstract'][0]['text']
        except :
            abstract =" "
        
        full_text =""
        for text in  j['body_text']:
            full_text += text['text']
        
        docs.append([paper_id,title,list_authors,abstract,full_text])

df = pd.DataFrame(docs,columns=['paper_id','title','list_authors','abstract','full_text'])
df.to_csv('/kaggle/working/data.csv')
df.head()

df['abstract_cleaned'] = df['abstract'].apply(lambda x: x.lower())
df['abstract_cleaned'] = df['abstract_cleaned'].apply(lambda x: x.translate(string.punctuation))
df['abstract_cleaned'] = df['abstract_cleaned'].apply(lambda x: x.translate(string.digits))

df['full_text_cleaned'] = df['full_text'].apply(lambda x: x.lower())
df['full_text_cleaned'] = df['full_text_cleaned'].apply(lambda x: x.translate(string.punctuation))
df['full_text_cleaned'] = df['full_text_cleaned'].apply(lambda x: x.translate(string.digits))

spacy_nlp = spacy.load('en_core_web_sm')

spacy_nlp.Defaults.stop_words |= {"et", "al", "novel", "data", "study", "studies", "research", "authors"}

spacy_stopwords = spacy_nlp.Defaults.stop_words

df['full_text_cleaned'] = df['full_text_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (spacy_stopwords)]))
df['abstract_cleaned'] = df['abstract_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (spacy_stopwords)]))
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(df['abstract_cleaned'], 20)
    
df2 = pd.DataFrame(common_words, columns = ['abstract_cleaned' , 'count'])
df2.groupby('abstract_cleaned').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in abstract')
common_words = get_top_n_words(df['full_text_cleaned'], 20)
    
df2 = pd.DataFrame(common_words, columns = ['full_text_cleaned' , 'count'])
df2.groupby('full_text_cleaned').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in full text')
df['abstract_word_count'] = df['abstract_cleaned'].apply(lambda x: len(x.strip().split()))
df['body_word_count'] = df['full_text_cleaned'].apply(lambda x: len(x.strip().split()))
df.head()
df.shape
df.isnull().sum()
# later in this work we found duplicates with differents id's we drop them by title
# we keep first as biorxiv papers has more information and abstractS
df = df.drop_duplicates(subset='title', keep='first')
for key in ['abstract_cleaned','title','full_text_cleaned']:
    total_words = df[key].values
    wordcloud = WordCloud(width=1800, height=1200).generate(str(total_words))
    plt.figure( figsize=(30,10) )
    plt.title ('Wordcloud' + key)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
df[['abstract_word_count']].plot(kind='box', title='Boxplot of Word Count', figsize=(10,6))
plt.show()

df[['body_word_count']].plot(kind='box', title='Boxplot of Word Count', figsize=(10,6))
plt.show()
print(df['abstract_word_count'].mean(),df['abstract_word_count'].std())
print(df['body_word_count'].mean(),df['body_word_count'].std())
features  = 5000
# TODO: probar con TFIDF
tf_vectorizer = CountVectorizer(max_features=features, stop_words='english', min_df=10)
X_tf = tf_vectorizer.fit_transform(df['abstract'])
tf_feat_name = tf_vectorizer.get_feature_names()
topics = 7
lda_model = LatentDirichletAllocation(learning_method='online',random_state=23,n_components=topics)
lda_output =lda_model.fit_transform(X_tf)
# preparing for plotting pyLDAvis
%matplotlib inline
pyLDAvis.enable_notebook()
# plotting lda
pyLDAvis.sklearn.prepare(lda_model, X_tf, tf_vectorizer)
# if you want to save it 
# P=pyLDAvis.sklearn.prepare(lda_model, X_tf, tf_vectorizer)
# pyLDAvis.save_html(p, 'lda.html')
# func from Anish Pandey
def visualizing_topic_cluster(model,stop_len,feat_name):
    topic={}
    for index,topix in enumerate(model.components_):
        topic[index]= [feat_name[i] for i in topix.argsort()[:-stop_len-1:-1]]
    
    return topic

topic_lda =visualizing_topic_cluster(lda_model,10,tf_feat_name)
# printing 
len([print('Topic '+str(key),topic_lda[key]) for key in topic_lda])
# dirty as fuck, removing words that appears in same topic
import copy
topic_lda_2 = topic_lda
for key in topic_lda_2:
    for element in topic_lda_2[key]:
        if element in ['cell','cells','viral','virus','respiratory','study','infection','acute']:
            topic_lda_2[key].remove(str(element))
[print('Topic '+str(key),topic_lda_2[key]) for key in topic_lda_2]
# lets see if our topics are correlated, first mixing topics with our df
columns=['Topic'+ i for i in list(map(str,list(topic_lda.keys())))]
lda_df =pd.DataFrame(lda_output,columns=columns).apply(lambda x : np.round(x,3))
lda_df['Major_topic'] =lda_df[columns].idxmax(axis=1).apply(lambda x: int(x[-1]))
lda_df['keyword'] = lda_df['Major_topic'].apply(lambda x: topic_lda[x])
lda_df.head()
# plotting results
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(abs(lda_df[columns].corr()),annot=True,fmt ='0.2f',cmap="YlGnBu")
plt.title(" Correlation Plot")
features  = 5000
tfidf_vectorizer = TfidfVectorizer(max_features=features, stop_words="english")
tfidf_vectorizer.fit(df['abstract'])
features_tf_idf = tfidf_vectorizer.transform(df['abstract'])
X = features_tf_idf
cls = MiniBatchKMeans(n_clusters=7, random_state=0)
Y = cls.fit_predict(X)
#Function for reducing (PCA) and visualizing cluster results
def reducing_and_visualizing_cluster_results(X,Y,centers):
    # reduce the features to 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)
    # reduce the cluster centers to 2D
    reduced_cluster_centers = pca.transform(centers)
    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=Y)
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=200, c='b')
    
reducing_and_visualizing_cluster_results(X.toarray(),Y,cls.cluster_centers_)
#function for training and creating word embedding model
def create_word_embedings(texts,method):
    documents=[]
    texts = clean_text(texts)
    for line in texts:
        documents.append(line.split())
    # build vocabulary and train model
    if(str(method)=="cbow"):
        model = Word2Vec(documents, size=50, window=5, min_count=5, workers=10)
    else:
        model = Word2Vec(documents, size=50, window=5, min_count=5, workers=10,sg=1)
    model.train(documents, total_examples=len(documents), epochs=10)
    return model
#function for cleaning and homogenizing the texts
def clean_text(text):
    text = text.apply(lambda x: re.sub('[^a-zA-Z0-9\s]', ' ', x))
    text = text.apply(lambda x: re.sub(' +',' ', x))
    text = text.apply(lambda x: x.lower())
    return text
texts = df['abstract']
model = create_word_embedings(texts,'cobw')
word_vectors = model.wv
del model
word_vectors.most_similar(positive = 'coronavirus', topn=10)
tokens = [token for token in word_vectors.vocab]
X = [word_vectors[token] for token in word_vectors.vocab]
(len(X),len(tokens))
silhouette_list = []
k_list = ['2','3','4','5','6','7','8','9','10','15','20','25','30']
for p in k_list:
    clusterer = AgglomerativeClustering(n_clusters=int(p), linkage='ward')
    clusterer.fit(X)
    # The higher (up to 1) the better
    s = round(metrics.silhouette_score(X, clusterer.labels_), 4)
    silhouette_list.append(s)
    # The higher (up to 1) the better

plt.figure(figsize=(10, 7))  
p = plt.plot(k_list,silhouette_list)
#plt.plot(['3','3'], [0,0.12], color="red", linewidth=2.5, linestyle="--")
plt.show()
clusterer = AgglomerativeClustering(n_clusters=5,  linkage='ward')
Y = clusterer.fit_predict(X)
cls_NC = NearestCentroid()
cls_NC.fit(X, Y)
reducing_and_visualizing_cluster_results(X,Y,cls_NC.centroids_)
labels = [word_vectors.similar_by_vector(center,topn = 5) for center in cls_NC.centroids_]
labels
# function returns WSS score for a list of k values
def calculate_WSS(points,func):
  sse = []
  k_list = ['2','3','4','5','6','7','8','9','10','15','20','25','30']
  module = importlib.import_module('sklearn.cluster')
  function = getattr(module, func)
  for k in k_list:
    kmeans = function(n_clusters = int(k)).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += np.linalg.norm(np.array(points[i])-np.array(curr_center)) ** 2
      
    sse.append(curr_sse)
  return sse
sse = calculate_WSS(X,'KMeans')
plt.figure(figsize=(10, 7))  
p = plt.plot(k_list,sse)
plt.show()
cls_KM =  KMeans(n_clusters=3, n_jobs=4, verbose=10)
cls_KM.fit(X)
Y = cls_KM.predict(X)
reducing_and_visualizing_cluster_results(X,Y,cls_KM.cluster_centers_)
labels = [word_vectors.similar_by_vector(center,topn = 5) for center in cls_KM.cluster_centers_]
labels
sse = calculate_WSS(X,'MiniBatchKMeans')
plt.figure(figsize=(10, 7))  
p = plt.plot(k_list,sse)
plt.show()
cls_MBKM = MiniBatchKMeans(n_clusters=3)
cls_MBKM.fit(X)
Y = cls_MBKM.predict(X)
reducing_and_visualizing_cluster_results(X,Y,cls_MBKM.cluster_centers_)
labels = [word_vectors.similar_by_vector(center,topn = 5) for center in cls_MBKM.cluster_centers_]
labels