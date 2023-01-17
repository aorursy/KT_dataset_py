# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install wordcloud
# Start with loading all necessary libraries

import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import seaborn as sns

import csv





import collections

print(os.listdir("../working/"))



import matplotlib.pyplot as plt

%matplotlib inline
# Load in the dataframe

df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
print("Number of rows in data =",df.shape[0])

print("Number of columns in data =",df.shape[1])

print("\n")

print("**Sample data:**")

df.head()

print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))



print("There are {} words in this dataset such as {}... \n".format(len(df.comment_text.unique()),

                                                                           ", ".join(df.comment_text.unique()[0:1])))

df[["comment_text"]].head()
categories = list(df.columns.values)

sns.set(font_scale = 2)

plt.figure(figsize=(15,8))

ax= sns.barplot(categories[2:], df.iloc[:,2:].sum().values)

plt.title("Comments in each category", fontsize=24)

plt.ylabel('Number of comments', fontsize=18)

plt.xlabel('Comment Type ', fontsize=18)

#adding the text labels

rects = ax.patches

labels = df.iloc[:,2:].sum().values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)

plt.show()
rowSums = df.iloc[:,2:].sum(axis=1)

multiLabel_counts = rowSums.value_counts()

multiLabel_counts = multiLabel_counts.iloc[1:]

sns.set(font_scale = 2)

plt.figure(figsize=(15,8))

ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)

plt.title("Comments having multiple labels ")

plt.ylabel('Number of comments', fontsize=18)

plt.xlabel('Number of labels', fontsize=18)

#adding the text labels

rects = ax.patches

labels = multiLabel_counts.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def clean(s): return re_tok.sub(r' \1 ', s)
import nltk

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import re

import sys

import warnings

data = df

if not sys.warnoptions:

    warnings.simplefilter("ignore")

def cleanHtml(sentence):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', str(sentence))

    return cleantext

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    cleaned = cleaned.strip()

    cleaned = cleaned.replace("\n"," ")

    return cleaned

def keepAlpha(sentence):

    alpha_sent = ""

    for word in sentence.split():

        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)

        alpha_sent += alpha_word

        alpha_sent += " "

    alpha_sent = alpha_sent.strip()

    return alpha_sent

data['comment_text'] = data['comment_text'].str.lower()

data['comment_text'] = data['comment_text'].apply(cleanHtml)

data['comment_text'] = data['comment_text'].apply(cleanPunc)

data['comment_text'] = data['comment_text'].apply(keepAlpha)
df = data.copy()
# Start with one review:

text = df.comment_text[0]



# Create and generate a word cloud image:

wordcloud = WordCloud(background_color="white").generate(text)

# Display the generated image:

plt.figure(figsize=(30,50))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
def prepare_text(text_col):

    ## decide vocab size

    text = text_col

    words = []

    for t in text:

        words.extend(tokenize(t))

    ##print(words[:100])

    vocab = list(set(words))

    ##print(len(words), len(vocab))

    words_str1 = ' '.join(str(e) for e in words)  

    

    # lower max_font_size, change the maximum number of word and lighten the background:

    wordcloud = WordCloud(stopwords=STOPWORDS,

                              collocations=False,

                              width=2500,

                              height=1800, background_color="white").generate(words_str1)

    plt.figure(figsize=(30,50))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()
prepare_text(df['comment_text'])
toxic_comments = df.loc[df.toxic != 0]['comment_text']

severe_toxic_comments = df.loc[df.severe_toxic != 0]['comment_text']

obscene_comments = df.loc[df.obscene != 0]['comment_text']

threat_comments = df.loc[df.threat != 0]['comment_text']

insult_comments = df.loc[df.insult != 0]['comment_text']

identity_hate_comments = df.loc[df.identity_hate != 0]['comment_text']
prepare_text(toxic_comments)
prepare_text(severe_toxic_comments)
prepare_text(obscene_comments)
prepare_text(threat_comments)
prepare_text(insult_comments)
prepare_text(identity_hate_comments)
import spacy

from spacy import displacy

nlp = spacy.load('en_core_web_sm')



from spacy.matcher import Matcher 

from spacy.tokens import Span 



import networkx as nx



from tqdm import tqdm
candidate_sentences = df['comment_text']

candidate_sentences.shape
doc = nlp("The 22-year-old recently won ATP Challenger tournament.")



for tok in doc:

  print(tok.text, "...", tok.dep_)
doc = nlp("Nagal won the first set.")



for tok in doc:

  print(tok.text, "...", tok.dep_)
doc = nlp("the drawdown process is governed by astm standard d823")



for tok in doc:

  print(tok.text, "...", tok.dep_)
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
get_entities("the film had 200 patents")
#entity_pairs = []



#for i in tqdm(candidate_sentences):

#  entity_pairs.append(get_entities(i))
## save paires

#filename = 'entity_pairs.csv'

#import csv

#with open(filename, 'w') as f:

#   writer = csv.writer(f, delimiter=',')

#   writer.writerows(entity_pairs)  #considering my_list is a list of lists.
## load paires 

l_entity_pairs = []

e_file = "../input/saved-relations/entity_pairs.csv" ## read preloaded entity_paires

#e_file = "hm_data/toxic_data/entity_pairs.csv" ## read session written entity paires

with open(e_file, 'r') as csvfile:

    entity_pairs_file = csv.reader(csvfile, delimiter=',')

    for row in entity_pairs_file:

        for re in row:

            re = re.replace('"','')

            re = eval(re)

            l_entity_pairs.append(re)      

entity_pairs = l_entity_pairs
entity_pairs[10:20]
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
 #relations = [get_relation(i) for i in tqdm(candidate_sentences)]
## save paires

#filename = 'relations.csv'

#import csv

#with open(filename, 'w') as f:

#   writer = csv.writer(f, delimiter=',')

#   writer.writerows(relations)  #considering my_list is a list of lists.

        
## load relations 

e_file = "../input/saved-relations/relations.csv" ## read preloaded entity_paires

#e_file = "hm_data/toxic_data/relations.csv" ## read session written entity paires

l_relations = []

with open(e_file, 'r') as csvfile:

    relations_file = csv.reader(csvfile, delimiter=',')

    for row in relations_file:

        for re in row:

            l_relations.append(re)

        #l_relations.append(''.join(row))
type(l_relations)
l_relations[1]
relations = l_relations
s = pd.Series(relations).value_counts()

s[:10]
## get as much as you want from verbs and their number of links

print(s[30:50])
# extract subject

source = [i[0] for i in entity_pairs]



# extract object

target = [i[1] for i in entity_pairs]



kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
# create a directed-graph from a dataframe

G=nx.from_pandas_edgelist(kg_df, "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())
#plt.figure(figsize=(12,12))



#pos = nx.spring_layout(G)

#nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)

#plt.show()
## incoming 

G=nx.from_pandas_edgelist(kg_df[kg_df['source']=="nigger"], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
## outgoing

G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="kiss"], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
## outging 

G=nx.from_pandas_edgelist(kg_df[kg_df['source']=="wikipedia"][:20], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5)

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
## incoming 

G=nx.from_pandas_edgelist(kg_df[kg_df['target']=="wikipedia"][:20], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5)

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="fuck"][:30], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="suck"][:25], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5)

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
## type at local computer 

# tensordboard --logdir model

## at bti_tf1 enviroment and project folder