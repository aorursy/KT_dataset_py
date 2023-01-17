# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import datetime

date_depart=datetime.datetime.now()

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import glob

import random

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import nltk

from nltk import ngrams



import spacy

import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer

from sklearn.decomposition import  PCA,NMF, LatentDirichletAllocation

from sklearn.decomposition import  IncrementalPCA

!python -m spacy download fr_core_news_md

!python -m spacy download fr_core_news_sm

import fr_core_news_md

nlp_fr = fr_core_news_md.load()

import  cloudpickle

import requests

import spacy.lang.fr

from stop_words import get_stop_words

stopwords_fr_set=set(nltk.corpus.stopwords.words('french'))

stopwords_fr_set.update(get_stop_words('fr'))

stopwords_fr_set.update(spacy.lang.fr.stop_words.STOP_WORDS)

stopwords_fr_set.update(["c'est","j'ai","n'est","n'ait","ca","ça","sais","jamais","chose","ex","'quelqu'",'quelqu'])

stopwords_fr_set.update((str(i) for i in range(30)))

stopwords_fr_set.update(["faut", "arrêter", "faisons", "faite", "faits",'oui' ,"www","https","http"])

stopwords_fr_set.update(requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.json").json())

stopwords_fr_set.update(str(i) for i in range(100))

stopwords_fr_set.update(str(i) for i in range(1980,2025))

stopwords_fr_set=list(stopwords_fr_set)

n_samples = 512*1024

vectorsamples=int(1e6)

n_features = 3000

n_components = 300

n_top_words = 15

ngram_range=(1,3)

max_df=0.6

min_df=5
tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,

                                   max_features=n_features,

                                   stop_words=stopwords_fr_set,

                                   ngram_range=ngram_range,

                                   dtype=np.float32

                                   )

tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df,

                                max_features=n_features,

                                stop_words=stopwords_fr_set,

                                ngram_range=ngram_range,

                                  dtype=np.uint16

                               )
texts=[]

texts_byfiles=dict()

filelist=glob.glob("../input/**/*.csv*", recursive=True)

random.shuffle(filelist)

for f in filelist:

    print (f)

    df=pd.read_csv(f,low_memory=False)

    dftext=[]

    texts_byfiles["f"]=dftext

    for n,s in df.items():

        for e in s:

            if isinstance(e,str):

                if len(e.split())>2 :

                    dftext.append(e)

    texts+=dftext



texts=list(set(texts))

random.shuffle(texts)

textes_base=texts
len(texts)
texts=random.sample(texts,n_samples)
tfidf_vectorizer.fit(random.sample(textes_base,vectorsamples))




tfidf = tfidf_vectorizer.transform(texts)

tfidf
rr = dict(zip(tfidf_vectorizer.get_feature_names(),  tfidf_vectorizer.idf_))



token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()

del rr

token_weight.columns=('token','weight')

token_weight = token_weight.sort_values(by='weight', ascending=False)

token_weight.reset_index(drop=True,inplace=True) 





sns.barplot(x='token', y='weight', data=token_weight.iloc[:60], )            

plt.title("Inverse Document Frequency(idf) per token")

fig=plt.gcf()

fig.set_size_inches(25,15)

ax=fig.axes[0]

plt.yscale("log")

ax.tick_params(axis='x',labelrotation=90 )

plt.show()



plt.figure(figsize=(12,12))

token_weight.plot()

# plt.yscale("log")

plt.show()
token_weight




nmf = NMF(n_components=n_components,

          alpha=.1, l1_ratio=.5,

           tol=5e-4,

          init="nndsvd",

          max_iter =20,

          shuffle =True,

          verbose=True)

nmf
tfidf_nmf=nmf.fit_transform(tfidf)

tfidf_nmf
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()

def get_top_words_list(model, feature_names, n_top_words):

    topwords=[]

    for topic_idx, topic in enumerate(model.components_):

       

        topwords.append([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

    return topwords






tfidf_feature_names = tfidf_vectorizer.get_feature_names()

print_top_words(nmf, tfidf_feature_names, n_top_words)

nmf_top_words_list=get_top_words_list(nmf, tfidf_feature_names, n_top_words)
ind_text=np.random.choice(n_samples,5)



for t,topics in zip([texts[i] for i in ind_text ],

                    

        tfidf_nmf[ind_text]):

    print(t)

    top_topics=np.argsort(topics)[-3:]

    for n in top_topics:

        print(f"topic {n}: {', '.join(nmf_top_words_list[n][:6])}")

    print()

    

pca_nmf = IncrementalPCA(n_components=3, batch_size=200)





tfidf_nmf_pca=pca_nmf.fit_transform(tfidf_nmf)





sns.scatterplot(x="pca1",y="pca2",hue="pca3" ,data=pd.DataFrame(tfidf_nmf_pca[np.random.choice(len(tfidf_nmf_pca),9000)]

                                                                ,columns=["pca1","pca2","pca3"]))

tf_vectorizer.fit(random.sample(textes_base,vectorsamples))


tf = tf_vectorizer.transform(texts)

tf
print(tf.max())


lda = LatentDirichletAllocation(n_components=n_components,

                                max_iter=20,

                                learning_method='online',

                                learning_offset=50.,

                              

                                verbose =1,

                                n_jobs =-1

                               )



lda.fit(tf)





partial_batch=32*1024

for i in range(0,len(textes_base),partial_batch):

    

    batchmat=tf_vectorizer.transform(textes_base[i:i+partial_batch])

    lda.partial_fit(batchmat)

    if (datetime.datetime.now()-date_depart)>datetime.timedelta(hours=7,minutes=20):

        break

    print(i)
tfidf_lda=lda.transform(tf)


print("\nTopics in LDA model:")

tf_feature_names = tf_vectorizer.get_feature_names()

lda_top_words_list=get_top_words_list(lda, tfidf_feature_names, n_top_words)

print_top_words(lda, tf_feature_names, n_top_words)
lda_best_topics=np.argsort(tfidf_lda.mean(axis=0))[-10:]

for n in range(6):

    print(f"topic {lda_best_topics[-n]}: {', '.join(lda_top_words_list[lda_best_topics[-n]][:15])}")

    

topic_texts=np.argsort(tfidf_lda,axis=0)[:300]

for t in range(tfidf_lda.shape[1]):

    for i in range(6):

        nmax=topic_texts[-i,t]

        print(tfidf_lda[nmax,t])

        print(texts[nmax][:300])    

    print(f"topic {t}: {', '.join(lda_top_words_list[t][:15])}")

    print("***")
ind_text=np.random.choice(len(textes_base),10)



for i in ind_text:

    t=textes_base[i]

    vect=tf_vectorizer.transform([t])

#     .toarray().flatten()

    topics=lda.transform(vect)

    

                    

        

    

    tf_vectorizer.transform(textes_base[i:i+partial_batch])

    print(t+"\n")

    top_topics=np.argsort(topics.flatten())[-3:]

    for n in top_topics:

        print(f"topic {n}: {', '.join(lda_top_words_list[n][:6])}")

    print("\n\n")
 


ipca = IncrementalPCA(n_components=3, batch_size=200)




for i in range(0,len(textes_base),partial_batch):

    batchmat=tf_vectorizer.transform(textes_base[i:i+partial_batch])

    batchmat_lda=lda.transform(batchmat)

    ipca.partial_fit(batchmat_lda)

    print(i,end="\r")

    


tfidf_lda_ipca=ipca.transform(tfidf_lda)

sns.scatterplot(x="pca1",y="pca2",hue="pca3" ,data=pd.DataFrame(tfidf_lda_ipca,columns=["pca1","pca2","pca3"]))