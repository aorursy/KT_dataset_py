#load library

import os

import pandas as pd

import numpy as np

import gensim

from gensim.utils import simple_preprocess

from gensim import corpora, models

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *



import datetime

import time

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



import nltk
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

print(meta.shape)
### first filter by meta file. select only papers after 2020

meta["publish_time"] = pd.to_datetime(meta["publish_time"])

meta["publish_year"] = (pd.DatetimeIndex(meta['publish_time']).year)

meta["publish_month"] = (pd.DatetimeIndex(meta['publish_time']).month)

meta = meta[meta["publish_year"] == 2020]

print(meta.shape[0], " papers are available after 2020 Jan 1.")
#count how many has abstract

count = 0

index = []

for i in range(len(meta)):

    #print(i)

    if type(meta.iloc[i, 8])== float:

        count += 1

    else:

        index.append(i)



print(len(index), " papers have abstract available.")
##extract the abstract to pandas 

documents = meta.iloc[index, 8]

documents=documents.reset_index()

documents.drop("index", inplace = True, axis = 1)



##create pandas data frame with all abstracts, use as input corpus

documents["index"] = documents.index.values

documents.head(3)
np.random.seed(400)

stemmer = SnowballStemmer("english")
##lemmatize and stemming



def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))



# Tokenize and lemmatize

def preprocess(text):

    result=[]

    for token in gensim.utils.simple_preprocess(text) :

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            # TODO: Apply lemmatize_stemming on the token, then add to the results list

            result.append(lemmatize_stemming(token))

    return result
## use example to check the preprocessing step



document_num = 1000  ##randomly pick one abstract

doc_sample = documents[documents["index"] == document_num].values[0][0]



print("Original document: ")

words = []

for word in doc_sample.split(' '):

    words.append(word)

print(words)

print("\n\nTokenized and lemmatized document: ")

print(preprocess(doc_sample))
##preprocess all abstracts

processed_docs = documents['abstract'].map(preprocess)

processed_docs[:5]
##create dictionary based on the preprocessed_documents

dictionary = gensim.corpora.Dictionary(processed_docs)



##check the dictionary

count = 0

for k, v in dictionary.iteritems():

    print(k, v)

    count += 1

    if count > 5:

        break
## remove extreme words (very common and very rare)

dictionary.filter_extremes(no_below=15, no_above=0.1)



##create bag-of-word model for each documents

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
## check the bow_corpus

bow_doc_1000 = bow_corpus[document_num]



for i in range(len(bow_doc_1000)):

    print("Word {} (\"{}\") appears {} time.".format(bow_doc_1000[i][0], 

                                                     dictionary[bow_doc_1000[i][0]], 

                                                     bow_doc_1000[i][1]))
#create tf-idf from bow_corpus

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]



#preview the corpus_tfidf

from pprint import pprint

for doc in corpus_tfidf:

    pprint(doc)

    break
now = datetime.datetime.now()

print ("start model building at ",now.strftime("%Y-%m-%d %H:%M:%S"))

lda_model = gensim.models.LdaMulticore(bow_corpus, 

                                       num_topics=5, 

                                       id2word = dictionary, 

                                       passes = 50, 

                                       workers=4) 



now = datetime.datetime.now()

print ('Model training finished at ',now.strftime("%Y-%m-%d %H:%M:%S"))
##print out the key words of five topics

for idx, topic in lda_model.print_topics(-1):

    print("Topic: {} \nWords: {}".format(idx, topic))

    print("\n")
now = datetime.datetime.now()

print ("start model building at ",now.strftime("%Y-%m-%d %H:%M:%S"))



lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, 

                                             num_topics=5, 

                                             id2word = dictionary, 

                                             passes = 50, 

                                             workers=4)

now = datetime.datetime.now()

print ('Model training finished at ',now.strftime("%Y-%m-%d %H:%M:%S"))
## check the key words of five topics

for idx, topic in lda_model_tfidf.print_topics(-1):

    print("Topic: {} Word: {}".format(idx, topic))

    print("\n")
documents_lda_topics = pd.DataFrame(columns = ["topic1", "topic2", "topic3", "topic4", "topic5"])

documents_lda_tfidf_topics = pd.DataFrame(columns = ["topic1", "topic2", "topic3", "topic4", "topic5"])

for i in range(len(bow_corpus)):

    if i % 500 ==0:

        print(i)

    documents_lda_topics.loc[i] = [0] * 5

    documents_lda_tfidf_topics.loc[i] = [0] * 5

    

    output = lda_model.get_document_topics(bow_corpus[i])

    for j in range(len(output)):

        a = output[j][0]

        b = output[j][1]

        documents_lda_topics.iloc[i,a] = b

    

    output_tfidf = lda_model_tfidf.get_document_topics(bow_corpus[i])

    for k in range(len(output_tfidf)):

        a = output_tfidf[k][0]

        b = output_tfidf[k][1]

        documents_lda_tfidf_topics.iloc[i, a] = b

        

print("Data processing finished")
## pick the final topic for each abstract based on max-probability

for i in range(5):

    documents_lda_topics.iloc[:, i] = documents_lda_topics.iloc[:, i].astype('float64', copy=False)

    

documents_lda_topics["final_topic"] =documents_lda_topics.iloc[:, :5].idxmax(axis=1)



for i in range(5):

    documents_lda_tfidf_topics.iloc[:, i] = documents_lda_tfidf_topics.iloc[:, i].astype('float64', copy=False)



documents_lda_tfidf_topics["final_topic"] =documents_lda_tfidf_topics.iloc[:, :5].idxmax(axis=1)
##preview the dataframe for both models

print("LDA + bow_corpus: topic probability:")

documents_lda_topics.head(3)

print("LDA + TF-IDF_corpus: topic probability:")

documents_lda_tfidf_topics.head(3)
pca = PCA(n_components=3)

pca_result = pca.fit_transform(documents_lda_topics.iloc[:, :5])
## with 3 components, variance explained

pca.explained_variance_ratio_
##create dataframe with projected vectors from PCA

pca_df = pd.DataFrame()

pca_df['pca-one'] = pca_result[:,0]

pca_df['pca-two'] = pca_result[:,1] 

pca_df["pca-three"] = pca_result[:, 2]

pca_df["topic"] = documents_lda_topics.iloc[:, 5].replace({"topic1": "red", "topic2": "blue", "topic3": "green", "topic4": "yellow", "topic5": "black"})
plt.figure(figsize=(16,10))

sns.scatterplot(

    x="pca-one", y="pca-two",

    hue= documents_lda_topics.iloc[:, 5].replace({"topic1": "red", "topic2": "blue", "topic3": "green", "topic4": "yellow", "topic5": "black"}),

    data=pca_df,

    legend="full",

    alpha=0.3)
ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(

    xs=pca_df["pca-one"], 

    ys=pca_df["pca-two"], 

    zs=pca_df["pca-three"], 

    cmap='tab10',

    c = documents_lda_topics.iloc[:, 5].replace({"topic1": "red", "topic2": "blue", "topic3": "green", "topic4": "yellow", "topic5": "black"})

)

ax.set_xlabel('pca-one')

ax.set_ylabel('pca-two')

ax.set_zlabel('pca-three')

plt.show()
##first run TSNE

import time

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(documents_lda_topics.iloc[:, :5])
##create dataframe with TSNE results

tsne_df = pd.DataFrame()

tsne_df['tsne-2d-one'] = tsne_results[:,0]

tsne_df['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    hue=documents_lda_topics.iloc[:, 5].replace({"topic1": "red", "topic2": "blue", "topic3": "green", "topic4": "yellow", "topic5": "black"}),

    #palette=sns.color_palette("hls", 10),

    data=tsne_df,

    legend="full",

    alpha=0.3)