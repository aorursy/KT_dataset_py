# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_json ('/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json',lines=True)
df.to_csv (r'output.csv', index = None)
df.head()
df.describe()
df.info()
df['headline'].shape
df.isnull().sum()
df.drop_duplicates('headline',keep = False, inplace = True)
df.head()
df.shape
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = df['headline'].values
vectorizer = TfidfVectorizer(stop_words = stop_words)
X = vectorizer.fit_transform(desc)
X
word_features = vectorizer.get_feature_names()
print(len(word_features))
print(word_features[7000:7300])
from sklearn.cluster import MiniBatchKMeans
def find_no_of_clusters(data, max_k):
    
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        value=MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data)
        sse.append(value.inertia_)           
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('Inertia')
    ax.set_title('Inertia vs Number of Clusters')
find_no_of_clusters(X, 20)
clusters = MiniBatchKMeans(n_clusters=14, init_size=1024, batch_size=2048, random_state=20).fit_predict(X)
def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
get_top_keywords(X, clusters, vectorizer.get_feature_names(), 10)
data= pd.read_json ('/kaggle/input/department-of-justice-20092018-press-releases/combined.json',lines=True)
data.to_csv (r'output.csv', index = None)
data.head()
data.info()
data.isnull().sum()
data.shape
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = data['contents'].values
vectorizer = TfidfVectorizer(stop_words = stop_words)
new_x = vectorizer.fit_transform(desc)
new_x
def find_no_of_clusters(data, max_k):
    
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        value=MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data)
        sse.append(value.inertia_)           
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('Inertia')
    ax.set_title('Inertia vs Number of Clusters')
    
find_no_of_clusters(new_x, 20)
clusters = MiniBatchKMeans(n_clusters=12, init_size=1024, batch_size=2048, random_state=20).fit_predict(new_x)
def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
get_top_keywords(new_x, clusters, vectorizer.get_feature_names(), 10)
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(new_x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()