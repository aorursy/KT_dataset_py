import pandas as pd

#load the data

data=pd.read_csv('../input/database-literature-review-crm-sust/orange_database_july2020.csv')
#check the loaded data

print(data.shape)
#look of the dataset

data.head()
data = data.fillna("")



#Separates only those who have not received sort and discard Duplicate or Rejected

data = data[data.status == 'Unclassified']
tech_databases = ['Tech_ACM',

 'Tech_Ebsco',

 'Tech_Emerald',

 'Tech_IEEE',

 'Tech_Jstor',

 'Tech_ScienceDirect']



sust_databases = ['Sust_Ebsco',

 'Sust_Emerald',

 'Sust_IEEE',

 'Sust_Jstor',

 'Sust_ScienceDirect']
tech = data[data['source'].isin(tech_databases)]



sust = data[data['source'].isin(sust_databases)]



tech = tech.reset_index(drop=True)

sust = sust.reset_index(drop=True)
#Let's look at how many articles we have per year from each area

for t in [tech,sust]:

    print(t.source.unique())

    print(t["year"].value_counts())

    print()
import re

import nltk

from nltk.corpus import stopwords



stop_words = set(stopwords.words('english')) 



def lowercase(text):

    text = str(text).lower()

    return text



def remove_stop_words(text, stopwords):

    for sw in stopwords:

        text = re.sub(r'\b%s\b' % sw, "", text)

        text = re.sub('\s+',' ',text).strip()

    return text



def remove_number(text):

    new_text = re.sub(r"[0-9]+", "", text)

    return new_text



def remove_punctuation(text):  

    # re.sub(replace_expression, replace_string, target)

    new_text = re.sub(r"\.|,|;|:|-|!|\?|´|`|^|'", " ", text)

    new_text = new_text.strip()

    return new_text



def remove_special(text):

    text = text.replace('highlights•','')

    text = text.replace('(','').replace(')','').replace('{','').replace('}','')

    text = text.replace('%','').replace('$','').replace('•','').replace('[','').replace(']','').replace('n/a','')

    return text



def preprocessing(df,fieldname):

    df[fieldname] = df[fieldname].apply(lowercase)

    df[fieldname] = df[fieldname].apply(remove_number)

    df[fieldname] = df[fieldname].apply(remove_punctuation)

    df[fieldname] = df[fieldname].apply(remove_stop_words,stopwords=stop_words)

    df[fieldname] = df[fieldname].apply(remove_special)
#Pre-processamento dos textos do abstract

preprocessing(tech,'abstract')



preprocessing(sust,'abstract')
#tfidf vector initililization

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()

tfidf = tfidf_vect.fit_transform(tech['abstract'].values)

tfidf.shape
from sklearn.cluster import KMeans

model_tf = KMeans(n_clusters = 10, n_jobs = -1,random_state=99)

model_tf.fit(tfidf)
labels_tf = model_tf.labels_

cluster_center_tf=model_tf.cluster_centers_
cluster_center_tf
# to understand what kind of words generated as columns by BOW

terms1 = tfidf_vect.get_feature_names()
terms1[1:10]
from sklearn import metrics

silhouette_score_tf = metrics.silhouette_score(tfidf, labels_tf, metric='euclidean')
silhouette_score_tf
# Giving Labels/assigning a cluster to each point/text 

df1 = tech

df1['Tfidf Clus Label'] = model_tf.labels_

df1.head(5)
# How many points belong to each cluster ->



df1.groupby(['Tfidf Clus Label'])['abstract'].count()
#Refrence credit - to find the top 10 features of cluster centriod

#https://stackoverflow.com/questions/47452119/kmean-clustering-top-terms-in-cluster

print("Top terms per cluster:")

order_centroids = model_tf.cluster_centers_.argsort()[:, ::-1]

for i in range(10):

    print("Cluster %d:" % i, end='')

    for ind in order_centroids[i, :10]:

        print(' %s' % terms1[ind], end='')

        print()
# visually how points or reviews are distributed across 10 clusters 

import matplotlib.pyplot as plt



plt.bar([x for x in range(10)], df1.groupby(['Tfidf Clus Label'])['abstract'].count(), alpha = 0.4)

plt.title('KMeans cluster points')

plt.xlabel("Cluster number")

plt.ylabel("Number of points")

plt.show()
# Reading a abstract which belong to each group.

for i in range(10):

    print("3 abstracts of assigned to cluster ", i)

    print("-" * 70)

    print(df1.iloc[df1.groupby(['Tfidf Clus Label']).groups[i][1]]['abstract'])

    print('\n')

    print(df1.iloc[df1.groupby(['Tfidf Clus Label']).groups[i][2]]['abstract'])

    print('\n')

    print(df1.iloc[df1.groupby(['Tfidf Clus Label']).groups[i][3]]['abstract'])

    print('\n')

    print("_" * 70)
# Train your own Word2Vec model using your own text corpus

i=0

list_of_sent=[]

for sent in tech['abstract'].values:

    list_of_sent.append(sent.split())
print(tech['abstract'].values[0])

print("*****************************************************************")

print(list_of_sent[0])
import gensim

# Training the wor2vec model using train dataset

w2v_model=gensim.models.Word2Vec(tech['abstract'],size=100, workers=4)
import numpy as np

sent_vectors = []; # the avg-w2v for each sentence/review is stored in this train

for sent in tech['abstract']: # for each review/sentence

    sent_vec = np.zeros(100) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        try:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

        except:

            pass

    sent_vec /= cnt_words

    sent_vectors.append(sent_vec)

sent_vectors = np.array(sent_vectors)

sent_vectors = np.nan_to_num(sent_vectors)

sent_vectors.shape

# Number of clusters to check.

num_clus = [3, 4, 5, 6, 7, 8, 9, 10]



# Choosing the best cluster using Elbow Method.

# source credit,few parts of min squred loss info is taken from different parts of the stakoverflow answers.

# this is used to understand to find the optimal clusters in differen way rather than used in BOW, TFIDF

squared_errors = []

for cluster in num_clus:

    kmeans = KMeans(n_clusters = cluster).fit(sent_vectors) # Train Cluster

    squared_errors.append(kmeans.inertia_) # Appending the squared loss obtained in the list

    

optimal_clusters = np.argmin(squared_errors) + 2 # As argmin return the index of minimum loss. 

plt.plot(num_clus, squared_errors)

plt.title("Elbow Curve to find the no. of clusters.")

plt.xlabel("Number of clusters.")

plt.ylabel("Squared Loss.")

xy = (optimal_clusters, min(squared_errors))

plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')

plt.show()



print ("The optimal number of clusters obtained is - ", optimal_clusters)

print ("The loss for optimal cluster is - ", min(squared_errors))
#tfidf vector initililization

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()

tfidf = tfidf_vect.fit_transform(sust['abstract'].values)

tfidf.shape
from sklearn.cluster import KMeans

model_tf = KMeans(n_clusters = 10, n_jobs = -1,random_state=99)

model_tf.fit(tfidf)
labels_tf = model_tf.labels_

cluster_center_tf=model_tf.cluster_centers_
cluster_center_tf
# to understand what kind of words generated as columns by BOW

terms1 = tfidf_vect.get_feature_names()
terms1[1:10]
from sklearn import metrics

silhouette_score_tf = metrics.silhouette_score(tfidf, labels_tf, metric='euclidean')
silhouette_score_tf
# Giving Labels/assigning a cluster to each point/text 

df2 = sust

df2['Tfidf Clus Label'] = model_tf.labels_

df2.head(5)
# How many points belong to each cluster ->



df2.groupby(['Tfidf Clus Label'])['abstract'].count()
#Refrence credit - to find the top 10 features of cluster centriod

#https://stackoverflow.com/questions/47452119/kmean-clustering-top-terms-in-cluster

print("Top terms per cluster:")

order_centroids = model_tf.cluster_centers_.argsort()[:, ::-1]

for i in range(10):

    print("Cluster %d:" % i, end='')

    for ind in order_centroids[i, :10]:

        print(' %s' % terms1[ind], end='')

        print()
# visually how points or reviews are distributed across 10 clusters 

import matplotlib.pyplot as plt



plt.bar([x for x in range(10)], df2.groupby(['Tfidf Clus Label'])['abstract'].count(), alpha = 0.4)

plt.title('KMeans cluster points')

plt.xlabel("Cluster number")

plt.ylabel("Number of points")

plt.show()
# Reading a abstract which belong to each group.

for i in range(10):

    print("3 abstracts of assigned to cluster ", i)

    print("-" * 70)

    print(df2.iloc[df2.groupby(['Tfidf Clus Label']).groups[i][1]]['abstract'])

    print('\n')

    print(df2.iloc[df2.groupby(['Tfidf Clus Label']).groups[i][2]]['abstract'])

    print('\n')

    print(df2.iloc[df2.groupby(['Tfidf Clus Label']).groups[i][3]]['abstract'])

    print('\n')

    print("_" * 70)
# Train your own Word2Vec model using your own text corpus

i=0

list_of_sent=[]

for sent in sust['abstract'].values:

    list_of_sent.append(sent.split())
print(sust['abstract'].values[0])

print("*****************************************************************")

print(list_of_sent[0])
# Training the wor2vec model using train dataset

w2v_model=gensim.models.Word2Vec(sust['abstract'],size=100, workers=4)
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this train

for sent in sust['abstract']: # for each review/sentence

    sent_vec = np.zeros(100) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        try:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

        except:

            pass

    sent_vec /= cnt_words

    sent_vectors.append(sent_vec)

sent_vectors = np.array(sent_vectors)

sent_vectors = np.nan_to_num(sent_vectors)

sent_vectors.shape
# Number of clusters to check.

num_clus = [3, 4, 5, 6, 7, 8, 9, 10]



# Choosing the best cluster using Elbow Method.

# source credit,few parts of min squred loss info is taken from different parts of the stakoverflow answers.

# this is used to understand to find the optimal clusters in differen way rather than used in BOW, TFIDF

squared_errors = []

for cluster in num_clus:

    kmeans = KMeans(n_clusters = cluster).fit(sent_vectors) # Train Cluster

    squared_errors.append(kmeans.inertia_) # Appending the squared loss obtained in the list

    

optimal_clusters = np.argmin(squared_errors) + 2 # As argmin return the index of minimum loss. 

plt.plot(num_clus, squared_errors)

plt.title("Elbow Curve to find the no. of clusters.")

plt.xlabel("Number of clusters.")

plt.ylabel("Squared Loss.")

xy = (optimal_clusters, min(squared_errors))

plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')

plt.show()



print ("The optimal number of clusters obtained is - ", optimal_clusters)

print ("The loss for optimal cluster is - ", min(squared_errors))
df_list = {'tech':df1,'sust':df2}



for key in df_list:

    for i in range(10):

        df = df_list[key]

        df = df[df['Tfidf Clus Label']==i]

        df.to_csv('{}_cluster_n{}.csv'.format(key,i))

        

print("Process Concluded!")