!pip install rake-nltk


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import WordNetLemmatizer,PorterStemmer

from nltk.tokenize import word_tokenize

from sklearn.cluster import DBSCAN

from nltk.corpus import stopwords

from spacy.matcher import Matcher 

from collections import  Counter

import matplotlib.pyplot as plt

from spacy.tokens import Span 

import tensorflow_hub as hub

from rake_nltk import Rake

import tensorflow as tf

import pyLDAvis.gensim

from tqdm import tqdm

import seaborn as sns

import networkx as nx

import pandas as pd

import numpy as np

import pyLDAvis

import gensim

import spacy

import os

import gc
path="../input/CORD-19-research-challenge/"

all_sources=pd.read_csv(path+"metadata.csv")
all_sources.isna().sum()
headline_length=all_sources['title'].str.len()

sns.distplot(headline_length)

plt.show()
headline_length=all_sources['abstract'].str.len()

plt.hist(headline_length)

plt.show()
stop=set(stopwords.words('english'))



def build_list(df,col="title"):

    corpus=[]

    lem=WordNetLemmatizer()

    stop=set(stopwords.words('english'))

    new= df[col].dropna().str.split()

    new=new.values.tolist()

    corpus=[lem.lemmatize(word.lower()) for i in new for word in i if(word) not in stop]

    

    return corpus
corpus=build_list(all_sources)

counter=Counter(corpus)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:10]:

    if (word not in stop) :

        x.append(word)

        y.append(count)
plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)
corpus=build_list(all_sources,"abstract")

counter=Counter(corpus)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:10]:

    if (word not in stop) :

        x.append(word)

        y.append(count)

        

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)
def get_top_ngram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:10]



top_n_bigrams=get_top_ngram(all_sources['title'].dropna(),2)[:10]

x,y=map(list,zip(*top_n_bigrams))

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)
top_tri_grams=get_top_ngram(all_sources['title'].dropna(),n=3)

x,y=map(list,zip(*top_tri_grams))

plt.figure(figsize=(9,7))

sns.barplot(x=y,y=x)
def preprocess_news(df):

    corpus=[]

    stem=PorterStemmer()

    lem=WordNetLemmatizer()

    for news in df['title'].dropna()[:5000]:

        words=[w for w in word_tokenize(news) if (w not in stop)]

        

        words=[lem.lemmatize(w) for w in words if len(w)>2]

        

        corpus.append(words)

    return corpus
corpus=preprocess_news(all_sources)

dic=gensim.corpora.Dictionary(corpus)

bow_corpus = [dic.doc2bow(doc) for doc in corpus]

lda_model =  gensim.models.LdaMulticore(bow_corpus, 

                                   num_topics = 4, 

                                   id2word = dic,                                    

                                   passes = 10,

                                   workers = 2)
lda_model.show_topics()


pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)

vis
del corpus,top_n_bigrams,lda_model,bow_corpus,top_tri_grams

gc.collect()


def prepare_similarity(vectors):

    similarity=cosine_similarity(vectors)

    return similarity



def get_top_similar(sentence, sentence_list, similarity_matrix, topN):

    # find the index of sentence in list

    index = sentence_list.index(sentence)

    # get the corresponding row in similarity matrix

    similarity_row = np.array(similarity_matrix[index, :])

    # get the indices of top similar

    indices = similarity_row.argsort()[-topN:][::-1]

    return [(i,sentence_list[i]) for i in indices]

module_url = "../input/universalsentenceencoderlarge4" 

# Import the Universal Sentence Encoder's TF Hub module

embed = hub.load(module_url)




titles=all_sources['title'].fillna("Unknown")

embed_vectors=embed(titles[:100].values)['outputs'].numpy()

sentence_list=titles.values.tolist()

sentence=titles.iloc[5]

print("Find similar research papers for :")

print(sentence)



similarity_matrix=prepare_similarity(embed_vectors)

similar=get_top_similar(sentence,sentence_list,similarity_matrix,6)



for sentence in similar:

    print(sentence)

    print("\n")
del embed_vectors,sentence_list,similarity_matrix

gc.collect()
nlp=spacy.load('en_core_web_sm')

sent_vecs={}

docs=[]



for i in tqdm(all_sources['title'].fillna('unknown')[:1000]):

    doc=nlp((i)) 

    docs.append(doc)

    sent_vecs.update({i :doc.vector})





sentences=list(sent_vecs.keys())

vectors=list(sent_vecs.values())

x=np.array(vectors)

dbscan=DBSCAN(eps=0.08, min_samples=2,metric='cosine' ).fit(x)

df_cluster=pd.DataFrame({'sentences':sentences,'label':dbscan.labels_})
df_cluster.label.unique()
df_cluster[(df_cluster['label']==0)].head()
df_cluster[(df_cluster['label']==1)].head()
path="../input/cord-19-eda-parse-json-and-generate-clean-csv/"

clean_comm=pd.read_csv(path+"clean_comm_use.csv",nrows=5000)

clean_comm['source']='clean_comm'

#clean_pmc=pd.read_csv(path+"clean_pmc.csv")

#clean_pmc['source']='clean_pmc'

biox = pd.read_csv(path+"biorxiv_clean.csv")

biox['source']='biorx'



all_articles=pd.concat([biox,clean_comm])
del biox,clean_comm

gc.collect()
all_articles.shape
tasks=["What is known about transmission, incubation, and environmental stability",

      "What do we know about COVID-19 risk factors",

      "What do we know about virus genetics, origin, and evolution",

      "What do we know about vaccines and therapeutics",

      "What do we know about non-pharmaceutical interventions",

      "What do we know about diagnostics and surveillance",

      "What has been published about ethical and social science considerations",

      "Role of the environment in transmission",

      "Range of incubation periods for the disease in humans",

      "Prevalence of asymptomatic shedding and transmission",

      "Seasonality of transmission",

      "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic)",

      "Susceptibility of populations",

      "Public health mitigation measures that could be effective for control",

      "Transmission dynamics of the virus",

      "Evidence that livestock could be infected",

      "Socioeconomic and behavioral risk factors for this spill-over",

      "Sustainable risk reduction strategies",

      "Resources to support skilled nursing facilities and long term care facilities",

      "Mobilization of surge medical staff to address shortages in overwhelmed communities"]
task_df=pd.DataFrame({'title':tasks,'source':'task'})
task_df.head()
all_articles=pd.concat([all_articles,task_df])

all_articles.fillna("Unknown",inplace=True)
sentence_list=all_articles.title.values.tolist()

embed_vectors=embed(sentence_list)['outputs'].numpy()

similarity_matrix=prepare_similarity(embed_vectors)

sentence= "Role of the environment in transmission"



similar=get_top_similar(sentence,sentence_list,similarity_matrix,10)
for sent in similar:

    print(sent[1])
ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])
import re

def clean(txt):

    txt=re.sub(r'\n','',txt)

    txt=re.sub(r'\([^()]*\)','',txt)

    txt=re.sub(r'https?:\S+\sdoi','',txt)

    return txt
texts=list(map(clean,texts))

text_list=' '.join(texts)

#text_list=word_tokenize(text_list)

!pip install python-rake
# Reka

import RAKE

import operator



# Reka setup with stopword directory

stop_dir = "../input/stopwordsforrake/SmartStoplist.txt"

rake_object = RAKE.Rake(stop_dir)



# Sample text to test RAKE





# Extract keywords

keywords = rake_object.run(text_list)

words,score=list(map(list,zip(*keywords)))

for word in (words[:10]):

    print(word)
!pip install pytextrank
import logging

import pytextrank

import spacy

import sys


nlp = spacy.load("en_core_web_sm")



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logger = logging.getLogger("PyTR")



# add PyTextRank into the spaCy pipeline



tr = pytextrank.TextRank(logger=None)

nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)



# parse the document



doc = nlp(text_list)



print("pipeline", nlp.pipe_names)

print("elapsed time: {} ms".format(tr.elapsed_time))





# examine the top-ranked phrases in the document



for phrase in doc._.phrases[:10]:

    print("{}".format(phrase.text))

    #print(phrase.chunks)
import spacy

nlp=spacy.load('en_core_web_sm')
def get_entities(sent):

    ## chunk 1

    ent1 = ""

    ent2 = ""



    prv_tok_dep = ""    # dependency tag of previous token in the sentence

    prv_tok_text = ""   # previous token in the sentence



    prefix = ""

    modifier = ""



  #############################################################

  

    for tok in nlp(sent):

        ## chunk 2

        # if token is a punctuation mark then move on to the next token

        if tok.dep_ != "punct":

          # check: token is a compound word or not

          if tok.dep_ == "compound":

            prefix = tok.text

            # if the previous word was also a 'compound' then add the current word to it

            if prv_tok_dep == "compound":

                   prefix = prv_tok_text + " "+ tok.text

      

      # check: token is a modifier or not

        if tok.dep_.endswith("mod") == True:

            modifier = tok.text

            # if the previous word was also a 'compound' then add the current word to it

            if prv_tok_dep == "compound":

              modifier = prv_tok_text + " "+ tok.text



          ## chunk 3

        if tok.dep_.find("subj") == True:

            ent1 = modifier +" "+ prefix + " "+ tok.text

            prefix = ""

            modifier = ""

            prv_tok_dep = ""

            prv_tok_text = ""      



          ## chunk 4

        if tok.dep_.find("obj") == True:

            ent2 = modifier +" "+ prefix +" "+ tok.text



          ## chunk 5  

          # update variables

        prv_tok_dep = tok.dep_

        prv_tok_text = tok.text

  #############################################################



    return [ent1.strip(), ent2.strip()]
def get_relation(sent):



  doc = nlp(sent)



  # Matcher class object 

  matcher = Matcher(nlp.vocab)



  #define the pattern 

  pattern = [{'DEP':'ROOT'}, 

            {'DEP':'prep','OP':"?"},

            {'DEP':'agent','OP':"?"},  

            {'POS':'ADJ','OP':"?"}] 



  matcher.add("matching_1", None, pattern) 



  matches = matcher(doc)

  k = len(matches) - 1



  span = doc[matches[k][1]:matches[k][2]] 



  return(span.text)
def prepare_df(text_list):

    doc=nlp(text_list)

    df=pd.DataFrame()

    for sent in list(doc.sents):

        sub,obj = get_entities(str(sent))

        relation= get_relation(str(sent))



        if ((len(relation)>2) & (len(sub)>2) &(len(obj)>2)):

            df=df.append({'subject':sub,'relation':relation,'object':obj},ignore_index=True)



    return df
df = prepare_df(text_list[24:])

df.head()


def draw_kg(pairs,c1='red',c2='blue',c3='orange'):

    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',

            create_using=nx.MultiDiGraph())

  

    node_deg = nx.degree(k_graph)

    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)

    plt.figure(num=None, figsize=(50, 40), dpi=80)

    nx.draw_networkx(

        k_graph,

        node_size=[int(deg[1]) * 500 for deg in node_deg],

        arrowsize=20,

        linewidths=1.5,

        pos=layout,

        edge_color=c1,

        edgecolors=c2,

        node_color=c3,

        )

    labels = dict(zip(list(zip(pairs.subject, pairs.object)),

                  pairs['relation'].tolist()))

    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,

                                 font_color='red')

    plt.axis('off')

    plt.show()
draw_kg(df)
sentence= "What is known about transmission, incubation, and environmental stability"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)

ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])
texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df)
sentence= "What do we know about COVID-19 risk factors"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,8)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "What do we know about virus genetics, origin, and evolution"



similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "What do we know about vaccines and therapeutics"



similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "Role of the environment in transmission"



similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence="What do we know about non-pharmaceutical interventions"



similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "What has been published about ethical and social science considerations"



similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence="What do we know about diagnostics and surveillance"



similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence="Range of incubation periods for the disease in humans"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence="Role of the environment in transmission"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence="Seasonality of transmission"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence="Prevalence of asymptomatic shedding and transmission"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')




sentence="Susceptibility of populations"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)
df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence="Public health mitigation measures that could be effective for control"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)



df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "Transmission dynamics of the virus"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)



df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "Evidence that livestock could be infected"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)



df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "Socioeconomic and behavioral risk factors for this spill-over"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)



df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "Sustainable risk reduction strategies"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)



df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "Resources to support skilled nursing facilities and long term care facilities"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)



df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')
sentence= "Mobilization of surge medical staff to address shortages in overwhelmed communities"

similar=get_top_similar(sentence,sentence_list,similarity_matrix,15)



ind,title=list(map(list,zip(*similar)))

titles=[]

texts=[]

for i in ind:

    titles.append(all_articles.iloc[i]['title'])

    texts.append(all_articles.iloc[i]['abstract'])

    

texts=list(map(clean,texts))

text_list=' '.join(texts)



df = prepare_df(text_list)

draw_kg(df,c1='blue',c2='pink',c3='green')