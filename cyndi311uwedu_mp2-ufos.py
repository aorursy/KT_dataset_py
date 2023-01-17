# Importing in the needed libraries according to the tutorial

import numpy as np

import pandas as pd

import requests

import lxml.html as lh

import nltk

import re

import os

import codecs

from sklearn import feature_extraction

import mpld3



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
url='http://www.nuforc.org/webreports/ndxe202002.html'

#Create a handle, page, to handle the contents of the website

page = requests.get(url)

#Store the contents of the website under doc

doc = lh.fromstring(page.content)

#Parse data that are stored between <tr>..</tr> of HTML

tr_elements = doc.xpath('//tr')

#Check the length of the first 12 rows

[len(T) for T in tr_elements[:12]]
#Create empty list

col=[]

i=0

#For each row, store each first element (header) and an empty list

for t in tr_elements[0]:

    i+=1

    name=t.text_content()

    print (i,name)

    col.append((name,[]))
#Since out first row is the header, data is stored on the second row onwards

for j in range(1,len(tr_elements)):

    #T is our j'th row

    T=tr_elements[j]

    

    #If row is not of size 10, the //tr data is not from our table 

    if len(T)!=7:

        break

    

    #i is the index of our column

    i=0

    

    #Iterate through each element of the row

    for t in T.iterchildren():

        data=t.text_content() 

        #Check if row is empty

        if i>0:

        #Convert any numerical value to integers

            try:

                data=int(data)

            except:

                pass

        #Append the data to the empty list of the i'th column

        col[i][1].append(data)

        #Increment i for the next column

        i+=1

[len(C) for (title,C) in col]
Dict={title:column for (title,column) in col}

df=pd.DataFrame(Dict)

df.head(20)
# load nltk's English stopwords as variable called 'stopwords'

stopwords = nltk.corpus.stopwords.words('english')

print(stopwords[:10])
# load nltk's SnowballStemmer as variabled 'stemmer'

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed



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
totalvocab_stemmed = []

totalvocab_tokenized = []

for i in df['Summary']:

    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem

    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

    

    allwords_tokenized = tokenize_only(i)

    totalvocab_tokenized.extend(allwords_tokenized)
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
from sklearn.feature_extraction.text import TfidfVectorizer



#define vectorizer parameters

tfidf_vectorizer = TfidfVectorizer(max_df=.99, max_features=200000,

                                 min_df=0.01, stop_words='english',

                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))



%time tfidf_matrix = tfidf_vectorizer.fit_transform(df['Summary']) #fit the vectorizer to synopses



print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()

print(terms)
from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tfidf_matrix)
import json



from sklearn.cluster import KMeans



num_clusters = 4



km = KMeans(n_clusters=num_clusters)



%time km.fit(tfidf_matrix)



clusters = km.labels_.tolist()
from sklearn.externals import joblib





joblib.dump(km, 'doc_cluster.pkl')



km = joblib.load('doc_cluster.pkl')

clusters = km.labels_.tolist()

df['Cluster'] = clusters

df['Cluster'].value_counts()
from __future__ import print_function



#Setting these up later for the data visualization

cluster_colors = {}

cluster_names = {}





print("Top terms per cluster:")

print()

#sort cluster centers by proximity to centroid

order_centroids = km.cluster_centers_.argsort()[:, ::-1] 



for i in range(num_clusters):

    print("Cluster %d words:" % i, end='')

    

    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster

        word = vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0]

        #coding this part in to dynamically change the labels based on where the clusters end up

        if "madar" in word:

            cluster_names[i] = 'MADAR, home, hovering'

            cluster_colors[i] = '#5DB7DE'

        elif "starlink" in word:

            cluster_names[i] ='Starlink, satellites, line'

            cluster_colors[i] = '#FFF3B0'

        elif "white" in word:

            cluster_names[i] ='bright, white, lights'

            cluster_colors[i] = '#0B4F6C'

        elif "object" in word:

            cluster_names[i] ='object,sky,flying'

            cluster_colors[i] = '#D0CFEC'

        print(' %s' % word, end=',')

    print() #add whitespace

    print() #add whitespace 

print()

print()
import matplotlib.pyplot as plt

import matplotlib as mpl



from sklearn.manifold import MDS



MDS()



# convert two components as we're plotting points in a two-dimensional plane

# "precomputed" because we provide a distance matrix

# we will also specify `random_state` so the plot is reproducible.

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)



pos = mds.fit_transform(dist)  # shape (n_components, n_samples)



xs, ys = pos[:, 0], pos[:, 1]
#some ipython magic to show the matplotlib plots inline

%matplotlib inline 





#create data frame that has the result of the MDS plus the cluster numbers and titles

new_df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 



#group by cluster

groups = new_df.groupby('label')





# set up plot

fig, ax = plt.subplots(figsize=(17, 9)) # set size

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

ax.set_facecolor((0, 0, 0))

#iterate through groups to layer the plot

#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label

for name, group in groups:

    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, 

            label=cluster_names[name], color=cluster_colors[name], 

            mec='none')

    ax.set_aspect('auto')

    ax.tick_params(\

        axis= 'x',          # changes apply to the x-axis

        which='both',      # both major and minor ticks are affected

        bottom=False,      # ticks along the bottom edge are off

        top=False,         # ticks along the top edge are off

        labelbottom=False)

    ax.tick_params(\

        axis= 'y',         # changes apply to the y-axis

        which='both',      # both major and minor ticks are affected

        left=False,      # ticks along the bottom edge are off

        top=False,         # ticks along the top edge are off

        labelleft=False)

    

ax.legend(numpoints=1)  #show legend with only 1 point

 

plt.show() #show the plot
