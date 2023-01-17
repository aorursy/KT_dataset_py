import numpy as np

import pandas as pd

import os

import nltk

nltk.download('punkt')

import re

import networkx as nx

from termcolor import colored 

nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words=stopwords.words('english')

stop_words.extend(['abstract','background','summary','introduction'])
smoke=pd.read_csv("../input/subtopic/smokpaper.csv",sep=',')

df=smoke.loc[[0]]

df
## tokenize sentences

from nltk.tokenize import sent_tokenize

sentences = []

for s in df['abstract']:

    sentences.append(sent_tokenize(s))



sentences= [y for x in sentences for y in x] 

sentences[:2]
## word embed from glove 

word_embedings = {}

f= open('../input/glovewordembed/glove.6B.100d.txt',encoding='utf-8')



for line in f:

    values=line.split()

    word=values[0]

    coefs=np.asarray(values[1:],dtype='float32')

    word_embedings[word]= coefs

f.close()  
## remove special char

clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")



# make alphabets lowercase

clean_sentences = [s.lower() for s in clean_sentences]
clean_sentences[:2]
## remove stop words

def remove_stopwords(sen):

    sen_new = " ".join([i for i in sen if i not in stop_words])

    return sen_new

clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

clean_sentences[:2]
##sentence to vectors using word embeddings

sentence_vectors = []

for i in clean_sentences:

  if len(i) != 0:

    v = sum([word_embedings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)

  else:

    v = np.zeros((100,))

  sentence_vectors.append(v)
sentence_vectors[:1]
# similarity matrix

sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences)):

  for j in range(len(sentences)):

    if i != j:

      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
import networkx as nx



nx_graph = nx.from_numpy_array(sim_mat)

scores = nx.pagerank(nx_graph)

## each node represents a sentence

nx.draw(nx_graph,pos=nx.spring_layout(nx_graph),with_labels = True)

nx_graph
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')

summary=summary.rename(columns={0:'score',1:"sentence"})

summary
ranked= summary['sentence'].values.tolist()

#top 20

#for i in range(len(ranked)):

  #print(i+1,")", ranked[i], "\n")

 

#print(colored(list(smoke['abstract']) ,'green'))

#first=sentences[0]

#aa=ranked[:3]

#aa.append(first)

#'.'.join(aa)   ## the first sentence  may be usefull



print(colored(list(df['abstract']) ,'green'))

'.'.join(ranked[:3])
preg= pd.read_csv("../input/subtopic/pregnantpaper.csv",sep=',')

df2=preg.loc[[10]]

df2

from nltk.tokenize import sent_tokenize

sentences = []

for s in df2['abstract']:

    sentences.append(sent_tokenize(s))



sentences= [y for x in sentences for y in x]  



## remove special char

clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")



# make alphabets lowercase

clean_sentences = [s.lower() for s in clean_sentences]



def remove_stopwords(sen):

    sen_new = " ".join([i for i in sen if i not in stop_words])

    return sen_new



# remove stopwords from the sentences

clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]





sentence_vectors = []

for i in clean_sentences:

  if len(i) != 0:

    v = sum([word_embedings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)

  else:

    v = np.zeros((100,))

  sentence_vectors.append(v)



sim_mat = np.zeros([len(sentences), len(sentences)])



from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences)):

  for j in range(len(sentences)):

    if i != j:

      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]



nx_graph = nx.from_numpy_array(sim_mat)

scores = nx.pagerank(nx_graph)    



## ranking 

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')

summary=summary.rename(columns={0:'score',1:"text"})



## output

summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')

summary=summary.rename(columns={0:'score',1:"text"})



ranked= summary['text'].values.tolist()

#top 20

#for i in range(len(ranked)):

  #print(i+1,")", ranked[i], "\n")

#for i in range(4):    

    #print( ranked[i])

print(colored(list(df2['abstract']) ,'green'))

'.'.join(ranked[:3])
card= pd.read_csv("../input/subtopic/respiratory_cardio_paper.csv",sep=',')

df3=card.loc[[2]]

from nltk.tokenize import sent_tokenize

sentences = []

for s in df3['abstract']:

    sentences.append(sent_tokenize(s))



sentences= [y for x in sentences for y in x]  



## remove special char

clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")



# make alphabets lowercase

clean_sentences = [s.lower() for s in clean_sentences]



def remove_stopwords(sen):

    sen_new = " ".join([i for i in sen if i not in stop_words])

    return sen_new



# remove stopwords from the sentences

clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]





sentence_vectors = []

for i in clean_sentences:

  if len(i) != 0:

    v = sum([word_embedings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)

  else:

    v = np.zeros((100,))

  sentence_vectors.append(v)



sim_mat = np.zeros([len(sentences), len(sentences)])



from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences)):

  for j in range(len(sentences)):

    if i != j:

      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]



nx_graph = nx.from_numpy_array(sim_mat)

scores = nx.pagerank(nx_graph)    



## ranking 

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')

summary=summary.rename(columns={0:'score',1:"text"})



## output

summary=pd.DataFrame(ranked_sentences).drop_duplicates(subset=1,keep='first')

summary=summary.rename(columns={0:'score',1:"text"})



ranked= summary['text'].values.tolist()

#top 20

#for i in range(len(ranked)):

  #print(i+1,")", ranked[i], "\n")

#for i in range(4):    

    #print( ranked[i])

print(colored(list(df3['abstract']) ,'green'))

'.'.join(ranked[:3])