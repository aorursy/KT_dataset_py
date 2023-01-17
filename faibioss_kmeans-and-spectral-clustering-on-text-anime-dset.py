#Imports



#Computing

import numpy as np 

import matplotlib.pyplot as plt

import pandas as pd



#Clustering

from sklearn.cluster import KMeans,SpectralClustering

from yellowbrick.cluster import KElbowVisualizer



#Decomposition

from sklearn.decomposition import PCA, TruncatedSVD



#Text

import unicodedata, re, string

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud



#Default



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Read the dataset

df=pd.read_csv('/kaggle/input/anime-dataset/anime.csv')

df.head()
#Some basic information

df.info()
animes=df[['title','description']].dropna()

animes.head()

#animes.count()


def remove_non_ascii(words):

    """Remove non-ASCII characters from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        new_words.append(new_word)

    return new_words



def remove_len_2(words):

    """Remove all the words with len <= 2"""

    new_words = []

    for word in words:

        if len(word)<=2:

            pass

        else:

            new_words.append(word)

    return new_words



def to_lowercase(words):

    """Convert all characters to lowercase from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = word.lower()

        new_words.append(new_word)

    return new_words



def remove_punctuation(words):

    """Remove punctuation from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = re.sub(r'[^\w\s]', '', word)

        if new_word != '':

            new_words.append(new_word)

    return new_words



def remove_numbers(words):

    """Remove all interger occurrences in list of tokenized words with textual representation"""

    new_words = []

    for word in words:

        new_word = re.sub("\d+", "", word)

        if new_word != '':

            new_words.append(new_word)

    return new_words



def remove_stopwords(words):

    """Remove stop words from list of tokenized words"""

    new_words = []

    for word in words:

        if word not in stopwords.words('english'):

            new_words.append(word)

    return new_words



def stem_words(words):

    """Stem words in list of tokenized words"""

    stemmer = LancasterStemmer()

    stems = []

    for word in words:

        stem = stemmer.stem(word)

        stems.append(stem)

    return stems



def lemmatize_verbs(words):

    """Lemmatize verbs in list of tokenized words"""

    lemmatizer = WordNetLemmatizer()

    lemmas = []

    for word in words:

        lemma = lemmatizer.lemmatize(word, pos='v')

        lemmas.append(lemma)

    return lemmas



def normalize(words):

    words = remove_non_ascii(words)

    words = remove_len_2(words)

    words = to_lowercase(words)

    words = remove_punctuation(words)

    words = remove_numbers(words)

    words = remove_stopwords(words)

    return words
#Tokenize text

animes['Tokenized']=animes['description'].apply(nltk.word_tokenize)

animes.head()
#Normalize text

animes['Clean_text']=animes['Tokenized'].apply(lambda y: normalize(y))

animes['Clean_text'][:10]
"""Function to reconvert a tokenization into a single string

This is needed for the TfIdfVectorizer method """

def conv2str(y):  

     

    str1 = " "   

    return (str1.join(y)) 
animes['Clean_text1']=animes['Clean_text'].apply(lambda y: conv2str(y))

animes.head()
"""Apply the TF_idf vectorizer to get the sparse matrix of the TF_IDF process"""



vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(animes['Clean_text1'])

#First row and the element extracted are the same.



print(X)

print(X.toarray()[0][20871])
"""A simple view of the feature names"""

print(vectorizer.get_feature_names()[:10])

"""First cluster using KMeans and run the elbow visualizer to find the best number of clusters"""

modelKm = KMeans(random_state=12)

visualizer = KElbowVisualizer(modelKm, k=(1,12))



visualizer.fit(X)        # Fit the data to the visualizer

visualizer.show() 
"""First cluster using Spectral and run the elbow visualizer to find the best number of clusters"""

modelSc = SpectralClustering(random_state=5)

visualizer = KElbowVisualizer(modelSc, k=(1,12))



visualizer.fit(X)        # Fit the data to the visualizer

visualizer.show() 
"""Train the Kmeans with the best n of clusters"""

modelKm = KMeans(n_clusters=4,random_state=12)

modelKm.fit(X)

y_kmeans = modelKm.predict(X)



"""Dimensionality reduction used to plot in 2d representation"""

pc=TruncatedSVD(n_components=2)

X_new=pc.fit_transform(X)

centr=pc.transform(modelKm.cluster_centers_)



print(centr)

plt.scatter(X_new[:,0],X_new[:,1],c=y_kmeans, cmap='viridis')

plt.scatter(centr[:,0],centr[:,1],marker='X',alpha=0.5,color='red',s=1000)
modelSc = SpectralClustering(n_clusters=4, random_state=5)

y_spc=modelSc.fit_predict(X)





pc=TruncatedSVD(n_components=2)

X_new=pc.fit_transform(X)



plt.scatter(X_new[:,0],X_new[:,1],c=y_spc, cmap='viridis')

#Rebuild the clusters in pandas df.



animes['ClusterKmeans']=y_kmeans

animes['ClusterSpectral']=y_spc

animes.head()
#Extract text based on cluster 

clus0_text=animes[animes['ClusterKmeans']==0]

clus1_text=animes[animes['ClusterKmeans']==1]

clus2_text=animes[animes['ClusterKmeans']==2]

clus3_text=animes[animes['ClusterKmeans']==3]
plt.figure(figsize=(12,8))

word_cloud = WordCloud(background_color='black',max_font_size = 80).generate(" ".join(clus0_text['Clean_text1']))

plt.imshow(word_cloud)

plt.axis('off')

plt.show()
plt.figure(figsize=(12,8))

word_cloud = WordCloud(background_color='black',max_font_size = 80).generate(" ".join(clus1_text['Clean_text1']))

plt.imshow(word_cloud)

plt.axis('off')

plt.show()
plt.figure(figsize=(12,8))

word_cloud = WordCloud(background_color='black',max_font_size = 80).generate(" ".join(clus2_text['Clean_text1']))

plt.imshow(word_cloud)

plt.axis('off')

plt.show()
plt.figure(figsize=(12,8))

word_cloud = WordCloud(background_color='black',max_font_size = 80).generate(" ".join(clus3_text['Clean_text1']))

plt.imshow(word_cloud)

plt.axis('off')

plt.show()
#Extract text based on cluster 

clus0_text_sp=animes[animes['ClusterSpectral']==0]

clus1_text_sp=animes[animes['ClusterSpectral']==1]

clus2_text_sp=animes[animes['ClusterSpectral']==2]

clus3_text_sp=animes[animes['ClusterSpectral']==3]

plt.figure(figsize=(12,8))

plt.subplot(221)

plt.title('Cluster1')

word_cloud = WordCloud(background_color='black',max_font_size = 80).generate(" ".join(clus0_text_sp['Clean_text1']))

plt.imshow(word_cloud)

plt.axis('off')



plt.subplot(222)

plt.title('Cluster2')

word_cloud = WordCloud(background_color='black',max_font_size = 80).generate(" ".join(clus1_text_sp['Clean_text1']))

plt.imshow(word_cloud)

plt.axis('off')



plt.subplot(223)

plt.title('Cluster3')

word_cloud = WordCloud(background_color='black',max_font_size = 80).generate(" ".join(clus2_text_sp['Clean_text1']))

plt.imshow(word_cloud)

plt.axis('off')



plt.subplot(224)

plt.title('Cluster4')

word_cloud = WordCloud(background_color='black',max_font_size = 80).generate(" ".join(clus3_text_sp['Clean_text1']))

plt.imshow(word_cloud)

plt.axis('off')





plt.show()