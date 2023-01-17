# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv")



print("Total number of docs \t\t\t\t\t {}".format(len(df)))

print("Number of docs with abstracts null \t\t\t {}".format(len(df[df['abstract'].isnull()])))

print("number of documents with no full text \t\t\t {}".format(len(df[df['has_full_text']==False])))

print("number of docs with no abstract and no full text \t {}".format(len(df[(df['abstract'].isnull()) & (df['has_full_text']==False)])))

print("Total count based on sha \t\t\t\t {}".format(df['sha'].count()))

print("Number of doc with unique sha \t\t\t\t {}".format(len(df['sha'].unique())))

print("Number of doc's with no sha \t\t\t\t {}".format(len(df[df['sha'].isnull()])))
df=df[~df['abstract'].isnull()].reset_index(drop=True)

df=df[['sha','source_x','title','doi','abstract']]
import numpy as np 

import pandas as pd

import re

import os

import nltk

from nltk.tokenize import RegexpTokenizer

import gensim

import fasttext

import fasttext.util

from keras.preprocessing import text

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from nltk.cluster import KMeansClusterer

from sklearn.cluster import DBSCAN

import seaborn as sns

from sklearn.manifold import TSNE 

from sklearn.decomposition import PCA

from sklearn.cluster import OPTICS

import matplotlib.pyplot as plt
embedding_dim=300



for i in sorted(os.scandir('../input/fasttext-crawl-300d-2m'), key=lambda x: x.stat().st_size, reverse=True):

    print(i.path)

    

model = gensim.models.KeyedVectors.load_word2vec_format("/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec")

tokenizer = RegexpTokenizer(r'\w+')

#word_token_final_list_st =[]



def cleantext1(text):

    lean1Text = re.sub('\(.*?\)','',text)

    month = "(January|February|March|April|May|June|July|August|September|October|November|December)"

    lean1Text = re.sub('[I|i]n\s'+month+'[of\s]\d{4}','',lean1Text) #18 February 2020

    lean1Text = re.sub('\d{2,}\s'+month,'',lean1Text)

    lean1Text = re.sub('\d{2,}','',lean1Text)

    lean1Text = re.sub('BACKGROUND:',' ',lean1Text)

    lean1Text=re.sub('&ndash;',' days to ',lean1Text)

    lean1Text=re.sub('mid-',' ',lean1Text)

    lean1Text=re.sub('\s'+month+'\sof\s\d{4}',' ',lean1Text)

    lean1Text = re.sub('(\s\d{1,}\sof\s'+month+'\sof\s\d{4})', ' ', lean1Text)

    lean1Text = re.sub('(\sof\s'+month+'\sof\s\d{4})', ' ', lean1Text)

    lean1Text = re.sub('(\s'+month+'\sof\s\d{4})', ' ', lean1Text)

    lean1Text = re.sub('r\$[\s+]?[\d+]?[\.]?[\,]?[\d{1,}]?[\.]?[\d{1,}]?[\.]?[\,]?[\d{1,}]?',' ',lean1Text)

    lean1Text = re.sub('-',' ',lean1Text)

    lean1Text = re.sub('\(.*?\)',' ',lean1Text)

    lean1Text = re.sub('\s\d\.\s',' ',lean1Text)

    lean1Text = re.sub('\s\w\.\s',' ',lean1Text)

    lean1Text = re.sub('(http[s]?...\w*.?\w*\.?\w*.*?\s)',' ',lean1Text)

    lean1Text= re.sub('\.\.+' , '.',lean1Text)

    lean1Text= re.sub('\.+', '.', lean1Text)

    lean1Text= re.sub('\s\.+','.',lean1Text)

    lean1Text=re.sub('\s\.','.',lean1Text)

    lean1Text=re.sub('\s\.\s{1,}','.',lean1Text)

    lean1Text=re.sub('\s{2,}',' ',lean1Text)

    lean1Text=re.sub('\.{2,}',' ',lean1Text)

    lean1Text=re.sub('\s\.\s{1,}','.',lean1Text)

    lean1Text=re.sub('\.\s\.','.',lean1Text)

    lean1Text=re.sub('&ndash;',' days to ',lean1Text)

    return lean1Text.strip()

df['cleaned_tex']=[cleantext1(x) for x in df['abstract'].tolist()]

train,test= train_test_split(df, test_size=0.20, random_state=42)
tokenizer= text.Tokenizer(num_words=5000)

tokenizer.fit_on_texts(train['cleaned_tex'].tolist())

train_word_index = tokenizer.word_index

train_embedding_weights = np.zeros((len(train_word_index)+1, embedding_dim))

ele_to_pop=[]

for word,index in train_word_index.items():

    if word in model:

        train_embedding_weights[index,:]=model[word]

    else:

        train_embedding_weights[index,:]= np.zeros(embedding_dim)

        ele_to_pop.append(word)

train_embedding_weights= train_embedding_weights[~np.all(train_embedding_weights==0.0,axis=1)]



for name,key in list(train_word_index.items()):

    if name in ele_to_pop:

        del train_word_index[name]
def ml_unsupervised(method,feature_matrix,numb_cluster=2):

    if method=='kmeans_scikit':

        km = KMeans(n_clusters=numb_cluster,max_iter=100)

        km.fit(feature_matrix)

        clusters = km.labels_

        return km, clusters



    if method=='optics':

        db=OPTICS(min_samples=10)

        db.fit(feature_matrix)

        clusters= db.labels_

        return db,clusters



def tsne_reduction(feature_matrix, convert_to_array='yes'):

    tsne = TSNE(n_components=2, random_state=0)

    np.set_printoptions(suppress=True)

    if convert_to_array=='yes':

        feature_matrix = tsne.fit_transform(feature_matrix.toarray())

    else:

        feature_matrix =  tsne.fit_transform(feature_matrix)

    return feature_matrix



def pca_reduction(feature_matrix,convert_to_array='yes'):

    pca = PCA(n_components=2, random_state=0)

    if convert_to_array=='yes':  

        feature_matrix = pca.fit_transform(feature_matrix.toarray())

    else:

        feature_matrix = pca.fit_transform(feature_matrix)

    return feature_matrix
tsne_reduced_embedding_weights= tsne_reduction(train_embedding_weights,convert_to_array='no')

pca_reduced_embedding_weights= pca_reduction(train_embedding_weights,convert_to_array='no')
from sklearn.metrics import silhouette_score

for i in range(2,7):

    k_model,k_cluster= ml_unsupervised('kmeans_scikit',tsne_reduced_embedding_weights,numb_cluster=i)

    sil_coeff = silhouette_score(tsne_reduced_embedding_weights, k_cluster, metric='euclidean')

    print(sil_coeff)
from sklearn.metrics import silhouette_score

for i in range(2,7):

    k_model,k_cluster= ml_unsupervised('kmeans_scikit',pca_reduced_embedding_weights,numb_cluster=i)

    sil_coeff = silhouette_score(pca_reduced_embedding_weights, k_cluster, metric='euclidean')

    print(sil_coeff)
kmeans_scan_obj,kmemans_clusters = ml_unsupervised('kmeans_scikit',pca_reduced_embedding_weights)

kmeans_sc_ts_obj,kmemans_sc_ts_clusters = ml_unsupervised('kmeans_scikit',tsne_reduced_embedding_weights)
df['kmeans_scikit(pca)_cluster']=pd.Series(kmemans_clusters)

df['kmeans_scikit(tsne)_cluster']=pd.Series(kmemans_sc_ts_clusters)

df['x']=pd.Series(pca_reduced_embedding_weights[:,0])

df['y']=pd.Series(pca_reduced_embedding_weights[:,1])

df['x_n']=pd.Series(tsne_reduced_embedding_weights[:,0])

df['y_n']=pd.Series(tsne_reduced_embedding_weights[:,1])
import seaborn as sns

f,axes = plt.subplots(2,1,figsize=(12,12),sharex=False,sharey=False,squeeze=False)

sns.set()

centers1=np.array(kmeans_scan_obj.cluster_centers_)

centers2=np.array(kmeans_sc_ts_obj.cluster_centers_)

sns.scatterplot(data=df,x=df['x'],y=df['y'],hue=df['kmeans_scikit(pca)_cluster'],ax=axes[0,0])

sns.scatterplot(centers1[:,0], centers1[:,1], marker="x", color='black',ax=axes[0,0])

sns.scatterplot(data=df,x=df['x_n'],y=df['y_n'],hue=df['kmeans_scikit(tsne)_cluster'],ax=axes[1,0])

sns.scatterplot(centers2[:,0], centers2[:,1], marker="x", color='black',ax=axes[1,0])

plt.show()
print('By using the Kmeans-pca dimensionality reduction')

print('Number of abstracts in cluster 0 \t {}'.format(len(df[df['kmeans_scikit(pca)_cluster']==0])))

print('Number of abstracts in cluster 1 \t {}'.format(len(df[df['kmeans_scikit(pca)_cluster']==1])))

print('\nBy using the Kmeans-tsne dimensionality reduction')

print('Number of abstracts in cluster 0 \t {}'.format(len(df[df['kmeans_scikit(tsne)_cluster']==0])))

print('Number of abstracts in cluster 0 \t {}'.format(len(df[df['kmeans_scikit(tsne)_cluster']==1])))
# db=OPTICS(min_samples=50)

# db.fit(tsne_reduced_embedding_weights)

# clusters= db.labels_

# d=db_clusters.tolist()

# print(list(set(d)))