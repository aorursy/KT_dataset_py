# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import nltk
# Read metadata csv alone, to read abstracts of each research paper

metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

metadata.head()
metadata['abstract'][0]  # Printing one abstract to get an idea of the data present
# Filter out only the abstract from the dataframe

abstract_df = metadata['abstract'].copy()

abstract_df = pd.DataFrame(abstract_df)
abstract_df.dropna(inplace=True)
abstract_df.reset_index(drop=True,inplace=True)
abstract_df
tokenized_text=[]
for i in range(0,len(abstract_df['abstract'])):

    tokenized_text.append(nltk.word_tokenize(abstract_df['abstract'][i]))
lemmatizer=nltk.stem.WordNetLemmatizer()
lemmatized_text=[]
for i in range(0,len(abstract_df['abstract'])):

    lemm_arr=[]

    for j in range(0,len(tokenized_text[i])):

        lemm_arr.append(lemmatizer.lemmatize(tokenized_text[i][j]))

    lemmatized_text.append(lemm_arr)
lemmatized_text[0:5]
# Removal of stop words

stop_words=set(nltk.corpus.stopwords.words('english'))
abstract_text = []

for i in range(0,len(lemmatized_text)):

    abstract_text.append([x for x in lemmatized_text[i] if x not in stop_words])
abstract_text[0]
from gensim.models import Word2Vec
model=Word2Vec(abstract_text,min_count=1,size=500,workers=10)
def sentence_vectorizer(sentence,model):

    sentence_vector=[]

    word_count=0

    for word in sentence:

        if word_count == 0:

            sentence_vector=model[word]

        else:

            sentence_vector=np.add(sentence_vector,model[word])

        word_count+=1

    return np.asarray(sentence_vector)/word_count
X_abstract=[]

for abstract in abstract_text:

    X_abstract.append(sentence_vectorizer(abstract,model))
# USing Kmeans clustering

from sklearn.cluster import KMeans

clusters_count=4

kmeans_clusterer=KMeans(clusters_count)

kmeans_clusterer.fit(X_abstract)

clusters=kmeans_clusterer.labels_.tolist()
clusters[15]
import matplotlib.pyplot as plt
import umap

embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(X_abstract)



plt.scatter(embedding[:,0],embedding[:,1],c=clusters)

plt.show()
for index,abstract in enumerate(abstract_text):    

    print(str(clusters[index])+" : "+' '.join(abstract))
abstract_classified_df=abstract_df.copy()

abstract_classified_df['class'] = pd.DataFrame(clusters)

abstract_classified_df.to_csv("/kaggle/working/classified_abstracts.csv")