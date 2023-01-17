import numpy as np

from numpy import array

import pandas as pd

import re

import string as str



import urllib.request

from bs4 import BeautifulSoup



import spacy

import nltk

from spacy.matcher import Matcher 

from spacy.tokens import Span 

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

from nltk.chunk import conlltags2tree, tree2conlltags



from nltk.corpus import stopwords                   # Stopwords corpus

from nltk.stem import PorterStemmer                 # Stemmer

from nltk.stem import WordNetLemmatizer             # WordNet



from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words

from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF

from gensim.models import Word2Vec                                   #For Word2Vec



from pprint import pprint

from collections import Counter

from tqdm import tqdm



import networkx as nx

import matplotlib.pyplot as plt
def url_to_string(url):

    #res = requests.get(url)

    #html = res.text

    html = urllib.request.urlopen(url)

    

    soup = BeautifulSoup(html, 'html.parser')

    

    # kill all script, style and other elements

    for script in soup(['script', 'style', 'button', 'a']):

        script.extract()



    # get text

    text = soup.get_text()

        

    # break into lines and remove leading and trailing space on each

    lines = [line.strip() for line in text.splitlines()]

    # break multi-headlines into a line each

    chunks = [phrase.strip() for line in lines for phrase in line.split("  ")]

    # drop blank lines

    text = '\n'.join([chunk for chunk in chunks if chunk])

    

    return text

url = 'https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news'

text = url_to_string(url)

print(text)
df = pd.DataFrame({'Sentences':text.split('\n')})

df
train_df = df[44:106]

train_df.reset_index(drop=True, inplace=True)

train_df
# list of sentences

segment = []

for text in train_df['Sentences']:

    segment.append(re.split('[^0-9]["."][^0-9]', text))



sentences = [sent for sents in segment for sent in sents]

#sentences.sort()

sentences
# Tune some sentences in the passage

sentences[13] = ' '.join(sentences[13:16])

sentences.pop(15)

sentences.pop(14)

sentences
def preprocess(sentences):

    list_of_words = []

    for sent in sentences:

        # Convert all words to lowercase

        sent = sent.lower()



        # Remove numbers

        # sent = re.sub(r'\d+', '', sent)



        # Remove punctuation

        #sent = sent.translate(string.maketrans(','), string.punctuation)



        # Remove space

        sent = sent.strip()



        # Remove stop-words and Stemming

        stop_words = set(stopwords.words('english'))

        tokens = nltk.word_tokenize(sent)

        tokens_ = [i for i in tokens if not i in stop_words]

        

        stemmer = PorterStemmer()

        for token in tokens_:

            stems = stemmer.stem(token)

            list_of_words.append(stems)



        # Lemmatization with WordNet

    #    lemmatizer = WordNetLemmatizer()

    #    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]

    

        #list_of_words.append(words)



    return list_of_words

list_of_words = preprocess(sentences)

list_of_words
# POS tagging (input: list of words)

POS = nltk.pos_tag(list_of_words)



# Chunking

pattern = 'NP: {<DT>?<JJ>*<NN>}'



NPChunker = nltk.RegexpParser(pattern)

result = NPChunker.parse(POS)

#print(result)

#result.draw()
# Named Entities Recognition

ne_tree = nltk.ne_chunk(POS)

print(ne_tree)
iob_tagged = tree2conlltags(result)

pprint(iob_tagged[:10])
# Run NLP model

#spacy_nlp = spacy.load('en')

nlp = spacy.load('en_core_web_sm') # load model



# Merge sentences to passage

passage = '.'.join(sentences)



# create a spaCy object 

doc = nlp(passage)



# print token, dependency, POS tag 

#for tok in doc:

#    print(tok.text, "-->",tok.dep_,"-->", tok.pos_)
# Type of each entities

for element in doc.ents:

    print('Type: %s, Value: %s' % (element.label_, element))
for element in doc.ents:

    print('Type: %s, Value: %s' % (element.label_, element))
#pprint([(X.text, X.label_) for X in doc.ents])
#pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])

# B = begins an entity, I = inside an entity, O = outside the entity
labels = [x.label_ for x in doc.ents]

Counter(labels)
items = [x.text for x in doc.ents]

Counter(items).most_common(3)
#spacy.displacy.render(nlp(sentences[50]), jupyter=True, style='ent')
spacy.displacy.render(nlp(sentences[50]), style='dep', jupyter = True, options = {'distance': 120})
[(x.orth_,x.pos_, x.lemma_) for x in [y 

                                      for y

                                      in nlp(sentences[50]) 

                                      if not y.is_stop and y.pos_ != 'PUNCT']]
spacy.displacy.render(doc, jupyter=True, style='ent')
# Simple knowledge graph by getting 2 entities

def get_entities(sent):

    ## chunk 1

    ent1 = ""

    ent2 = ""



    prv_tok_dep = ""    # dependency tag of previous token in the sentence

    prv_tok_text = ""   # previous token in the sentence



    prefix = ""

    modifier = ""

  

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



    return [ent1.strip(), ent2.strip()]
entity_pairs = []



for i in sentences:

    entity_pairs.append(get_entities(i))
entity_pairs
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
relations = [get_relation(i) for i in tqdm(sentences)]
pd.Series(relations).value_counts()[:50]
# extract subject

source = [i[0] for i in entity_pairs]



# extract object

target = [i[1] for i in entity_pairs]



kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
# create a directed-graph from a dataframe

G=nx.from_pandas_edgelist(kg_df, "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(12,12))



pos = nx.spring_layout(G)

nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="accepts"], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()