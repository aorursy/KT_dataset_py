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
#load libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import nltk,string,re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import gensim
from gensim.models import Word2Vec 
import warnings
warnings.filterwarnings("ignore")
#load the data
import pandas as pd
df=pd.read_csv('../input/cbc-news-coronavirus-articles-march-26/news.csv',encoding='utf-8')
df.describe(include='all')
#check null values
df.isnull().sum()
#clean authors from punctuations
df['authors']=df['authors'].astype(str)
df['authors']=df['authors'].apply(lambda x: str(x).replace(string.punctuation, ''))
len(df.authors.unique())
sns.countplot(df['publish_date'].astype('datetime64').dt.month)
sns.countplot(df['publish_date'].astype('datetime64').dt.year)
#append these three fields in one field
df['article']= df['title'].astype(str) +' '+ df['description'].astype(str) +' '+df['text'].astype(str)
print('before preprocessing')
print(df['article'][0])

#tokenize articles to sentences
df['article']=df['article'].apply(lambda x: nltk.sent_tokenize(x))

#tokenize articles sentences to words
df['article']=df['article'].apply(lambda x: [nltk.word_tokenize(sent) for sent in x])

#lower case
df['article']=df['article'].apply(lambda x: [[wrd.lower() for wrd in sent] for sent in x])

#White spaces removal
df['article']=df['article'].apply(lambda x: [[wrd.strip() for wrd in sent if wrd != ' '] for sent in x])

#remove stop words 
stopwrds = set(stopwords.words('english'))
df['article']=df['article'].apply(lambda x: [[wrd for wrd in sent if not wrd in stopwrds] for sent in x])

#remove punctuation words
table = str.maketrans('', '', string.punctuation)
df['article']=df['article'].apply(lambda x: [[wrd.translate(table) for wrd in sent] for sent in x])

#remove not alphabetic characters
df['article']=df['article'].apply(lambda x: [[wrd for wrd in sent if wrd.isalpha()] for sent in x])

#lemmatizing article 
lemmatizer = WordNetLemmatizer()
df['article']=df['article'].apply(lambda x:[[lemmatizer.lemmatize(wrd.strip()) for wrd in sent ] for sent in x ])

#remove single characters
df['article']=df['article'].apply(lambda x: [[wrd for wrd in sent if len(wrd)>2] for sent in x])

#reformat article column to single text not nested lists
df['article']=df['article'].apply(lambda x:[' '.join(wrd) for wrd in x])
df['article']=df['article'].apply(lambda x:' '.join(x))

print('\n')
print('after preprocessing')
print(df['article'][0])
#TF IDF for article column
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
tfidf_article=tfidf_vectorizer.fit_transform(df['article'])
#tf-idf output vectors
from sklearn.decomposition import PCA 
tfidf_pca = PCA(n_components = 2) 
tfidf_pca_comp = tfidf_pca.fit_transform(tfidf_article.toarray())
tfidf_pca_comp.shape
clusters=[]
kmeans_scores=[]
from sklearn.cluster import KMeans
K = range(2, 20)
for k in K:
    k_means = KMeans(n_clusters=k)
    k_means.fit(tfidf_pca_comp)
    kmeans_scores.append(k_means.score(tfidf_pca_comp))
    clusters.append(k)
plt.scatter(clusters,kmeans_scores)
plt.xlabel("No. of clusters")
plt.ylabel("scores")
plt.show()
tfidf_pca_comp.shape
k_means = KMeans(n_clusters=5)
k_means.fit(tfidf_pca_comp)
pred=k_means.predict(tfidf_pca_comp)
plt.figure(figsize=(20,20))
plt.scatter(tfidf_pca_comp[:,0],tfidf_pca_comp[:,1],c=pred)
plt.show()
df['tfidf']=tfidf_article
df['tfidf_clusters']=pred
df.head()
tfidf_article.shape
top_tf_df = pd.DataFrame(tfidf_article.todense()).groupby(df['tfidf_clusters']).mean()
top_tf_df
for i,r in top_tf_df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([tfidf_vectorizer.get_feature_names()[t] for t in np.argsort(r)[-20:]]))