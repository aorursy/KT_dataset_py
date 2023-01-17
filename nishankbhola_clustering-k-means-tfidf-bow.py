# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Creating dataframe

df = pd.read_csv("../input/amazon-fine-food-reviews/Reviews.csv")

df.head(2)
sns.countplot(df.Score)
#Dropping "Na" rows

df = df.dropna()



#we total "568411" reviews
#re-indexing the dataframe due to missing rows 

df = df.reset_index(drop=True)

df.shape
#taking first 10000 reviews due to computational power

new_df = df.head(100000)

new_df.shape
sns.countplot(new_df.Score)
#Data Cleaning

# removing html tags , punctuations , stopwords and applying lemmatizer

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer





remove_words = ['com','http','https','www']

corpus =[]

for i in range(0,len(new_df['Text'])):

    if (i == 25509):

        pass

    #print(i)

    review = re.sub("<.*?>", " ", new_df['Text'][i])

    review = re.sub('[^a-zA-Z]' , ' ' ,review)

    review = review.lower()

    review = review.split()

    lemmatizer = WordNetLemmatizer()

    review = [lemmatizer.lemmatize(words) for words in review if not words in set(stopwords.words('english'))] 

    review = [word for word in review if word not in remove_words]

    review = ' '.join(review)

    corpus.append(review)

corpus[0]

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(max_features=1500)

bow = tfidf_vec.fit_transform(corpus)

bow.shape

terms = tfidf_vec.get_feature_names()

terms[1:10]
# Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans

clusters_list = [2,25,50,100,150,200]

wcss = []

for i in clusters_list:

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(bow)

    wcss.append(kmeans.inertia_)
plt.plot(clusters_list, wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 25, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(bow)

labels_tf = kmeans.labels_

cluster_center_tf=kmeans.cluster_centers_

cluster_center_tf
#find the top 10 features of cluster centriod

print("Top terms per cluster:")

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(25):

    print("Cluster %d:" % i, end='')

    for ind in order_centroids[i, :9]:

        print(' %s' % terms[ind], end='')

        print()
new_df.shape
#converting list to dataframe

cluster_df = pd.DataFrame({'clusters':y_kmeans[:]})
#concatinating ckuster dataframe and review dataframe

new_df = pd.concat([new_df, cluster_df], axis=1, sort=False)

new_df.head()
# value count of each cluster

new_df['clusters'].value_counts()
#clusters 

sns.countplot(new_df['clusters'])
#reading review defined by each cluster

for i in range(25):

    #print("Cluster %d:" % i, end='')

    print("4 review of assigned to cluster ", i, end='\n')

    print("-" * 70)

    print(new_df.iloc[new_df.groupby(['clusters']).groups[i][5]]['Text'])

    print('\n')

    print(new_df.iloc[new_df.groupby(['clusters']).groups[i][10]]['Text'])

    print('\n')

    print(new_df.iloc[new_df.groupby(['clusters']).groups[i][20]]['Text'])

    print('\n')

    print("_" * 70)