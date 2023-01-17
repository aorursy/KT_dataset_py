import os
base_path="mycomputer\\Probabilistic Graphical Models\\Lab Exam\\Lab Exam"

os.chdir(base_path)


from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine


# Importing necessary library
import pandas as pd
import numpy as np
import nltk
import os
import nltk.corpus
# importing word_tokenize from nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk

import nltk

# uncomment and run when using first time
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('words')
#nltk.download('maxent_ne_chunker')



my_file = os.path.join(base_path + "/" + "panchatantra-tales_complete.pdf")
fp = open(my_file, 'rb')
parser = PDFParser(fp)
doc = PDFDocument(parser)
#parser.set_document(doc)
#doc.set_parser(parser)
#doc.initialize('')
rsrcmgr = PDFResourceManager()
laparams = LAParams()
laparams.char_margin = 1.0
laparams.word_margin = 1.0
device = PDFPageAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)

extracted_text = ''

for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
        layout = device.get_result()
        for lt_obj in layout:
            if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                extracted_text += lt_obj.get_text()
                
#print(extracted_text)
start =extracted_text.find("1.The Monkey And The Wedge")
end=extracted_text.find("2.The Jackal And The Drum")

print(start, end)
story= extracted_text[start:end]
print(story[0:500])
#printing limited characters
# Passing the string text into word tokenize for breaking the sentences
word_tokens = word_tokenize(story)
#print(word_tokens)
# finding the frequency distinct in the tokens
# Importing FreqDist library from nltk and passing token into FreqDist
fdist = FreqDist(word_tokens)
fdist
# To find the frequency of top 10 words
fdist1 = fdist.most_common(10)
fdist1
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))

filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
print(word_tokens[0:30]) 
print(filtered_sentence[0:30]) 

#POS Tagging
for tex in word_tokens:
    pos_tag=nltk.pos_tag([tex])
    #print(pos_tag)

## for brevity, printing only limited number of POS_tag as below, otherwise upper loop prints all POS
for tex in word_tokens[0:20]:
    pos_tag=nltk.pos_tag([tex])
    print(pos_tag)
from nltk import CFG
groucho_grammar = CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")
# ref: http://www.nltk.org/book_1ed/ch08.html

sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
parser = nltk.ChartParser(groucho_grammar)
trees = parser.parse(sent)
for tree in trees:
    print (tree)
## part C begins here
#conda install -c conda-forge spacy=2.2.1
# conda install -c conda-forge spacy-model-en_core_web_sm
#https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/
import spacy
print(spacy.__version__)

### extracting dependency parsing because POS tagging is not sufficient many times
nlp = spacy.load('en_core_web_sm')

#Example
doc = nlp("The 22-year-old recently won ATP Challenger tournament.")

for tok in doc:
  print(tok.text, "...", tok.dep_)
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)
%matplotlib inline

doc=nlp(story)
# for bravity, printing only limited output
for tok in doc[0:20]:
  print(tok.text, "...", tok.dep_)
#The main idea is to go through a sentence and extract the subject and the object as and when they are encountered so that we can
# have nodes and edges for the graph

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

from nltk import sent_tokenize

sentence=sent_tokenize(story)
print (sentence[0])
entity_pairs = []

for i in tqdm(sentence):
    entity_pairs.append(get_entities(i))
entity_pairs[10:20]
#Our hypothesis is that the predicate is actually the main verb in a sentence.
#The function below is capable of capturing such predicates from the sentences.
#The pattern defined in the function tries to find the ROOT word or the main verb in the sentence. 
#Once the ROOT is identified, then the pattern checks whether it is followed by a preposition (‘prep’) 
#or an agent word. If yes, then it is added to the ROOT word.


# some patterns and their meaning

#GDP --> nsubj --> NOUN
#in --> prep --> ADP
#developing --> amod --> VERB
#countries --> pobj --> NOUN
#such --> amod --> ADJ
#as --> prep --> ADP
#Vietnam --> pobj --> PROPN
#will --> aux --> VERB
#continue --> ROOT --> VERB
#growing --> xcomp --> VERB
#at --> prep --> ADP
#a --> det --> DET
#high --> amod --> ADJ
#rate --> pobj --> NOUN
#. --> punct --> PUNCT

def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object 
  matcher = Matcher(nlp.vocab)

  #define the pattern 
  pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"},
            {"POS": "ADV", "OP": "*"},
            {'DEP':'amod', 'OP':"?"}] 

  matcher.add("matching_1", None, pattern) 

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]] 

  return(span.text)
relations = [get_relation(i) for i in tqdm(sentence)]
print(relations)
#most frequent relations or predicates that we have just extracted:
pd.Series(relations).value_counts()[:10]
total= len(relations)
total
### joint probability table
prob_table = pd.Series(relations).value_counts()[:10]/ total
print(prob_table)
#we create a dataframe of entities and predicates:

# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
#print(source[1:10])
#print(target[1:10])

print(kg_df)
# create a directed-graph from a dataframe
G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="told"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k=0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=2500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()