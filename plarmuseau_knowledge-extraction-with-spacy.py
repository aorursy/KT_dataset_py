import numpy as np

import pandas as pd

import os

import json

import glob

import sys

import argparse

import codecs

import logging

import time



import tqdm

sys.path.insert(0, "../")



root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%%time

import fnmatch



ABS = pd.DataFrame(columns = ["Title", "Abstract", "URL"])



df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

df=df[["title", "cord_uid", "sha", "url"]]



### Scrapping to retreive relevant articles ####



for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            if fnmatch.fnmatch(filename, '*.json'):

                path = os.path.join(dirname, filename)

                data = json.load(open(path))

                paper_id = data["paper_id"]                

            

                try: 

                    abst = pd.DataFrame(data["abstract"])

                    abst =abst[["text"]]

                    abst=abst.rename(columns={"text": "Abstract"})



                    art = pd.DataFrame(data["metadata"])

                    art =art[["title"]]

                    art=art.rename(columns={"title": "Title"})



                    art["Abstract"] = abst.iloc[0,0]

                    subset = df[df["sha"]==paper_id]

                    art["URL"] = subset.iloc[0,3]



                    #### Track articles that explicitly mention coronaviruses and close derivatives in their abstracts ####



                    Covid_List = ["COVID", "covid", "Covid", "coronavirus", \

                                  "Coronavirus", "2019-nCov",  "SARS-CoV-2",'pathway interleukin-6','Chloroquine ','Selenium','Ginseng','Glycyrrhizic','Zinc','Nucleotide prodrug','mRNA vaccine neutralizing antibodies','CCR5 antagonist','inhibit the RNA-dependent RNA polymerase','ACE','bradykinin','icatibant','melatonin','lopinavir' ]                 

                    

                    ABS = pd.concat([ABS, art], sort = False)

                        

                except:

                    pass



            

ABS = ABS [["URL", "Title", "Abstract"]]

ABS = ABS.drop_duplicates()

ABS.to_csv("Relevant_COVID-19_Abstracts.csv", index = False)



print("Number of Relevant Abstracts", len(ABS))



#### Embedding Hyperlinks in the DataFrame ####



for i in range(len(ABS)):

    if ABS.iloc[i, 1] != "":

        ABS.iloc[i, 1] = '<a href="{}">{}</a>'.format(ABS.iloc[i, 0], ABS.iloc[i, 1])

    else:

        ABS.iloc[i, 1] = '<a href="{}">{}</a>'.format(ABS.iloc[i, 0], "Link to Article")



ABS = ABS[["Title", "Abstract"]]



ABS
import spacy

nlp = spacy.load('en_core_web_sm')



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
def get_entities(sent):

  ## chunk 1 The main idea is to go through a sentence and extract the subject and the object as and when they are encountered. 

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

get_relation(senti.text)
entity_pairs = []

relations=[]

for abstr in tqdm(ABS.Abstract):

    for senti in nlp(abstr).sents:

        entity_pairs.append(get_entities(senti.text))

        relations.append(get_relation(senti.text))



        
#most frequent relations or predicates that we have just extracted

pd.Series(relations).value_counts()[:50]
# extract subject

source = [i[0] for i in entity_pairs]



# extract object

target = [i[1] for i in entity_pairs]



kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

kg_df.to_csv('entitypairs.csv')
# create a directed-graph from a dataframe

G=nx.from_pandas_edgelist(kg_df[:100], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))



pos = nx.spring_layout(G)

nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
#Role of the environment in transmission

G=nx.from_pandas_edgelist(kg_df[kg_df.target=="environment"].append(kg_df[kg_df.source=="environment"]), "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
kg_df[kg_df.target=="lockdown"]
#Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings

G=nx.from_pandas_edgelist(kg_df[kg_df.target=="facemask"].append(kg_df[kg_df.target=="gloves"]).append(kg_df[kg_df.source=="effectiveness"]), "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
def zoek(woord):

    zoekdf=kg_df[kg_df.source==woord].append(kg_df[kg_df.target==woord].append(kg_df[kg_df.edge==woord]))

    #print(zoekdf)

    return zoekdf

#Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings



G=nx.from_pandas_edgelist(zoek('prevent'), "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
#Immune response and immunity

G=nx.from_pandas_edgelist(zoek('response').append(zoek('immunity') ) , "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
#Tools and studies to monitor phenotypic change and potential adaptation of the virus



G=nx.from_pandas_edgelist(zoek('phenotype').append(zoek('mutation') ) , "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
#Disease models, including animal models for infection, disease and transmission



G=nx.from_pandas_edgelist(zoek('modelling').append(zoek('epidemiology') ) , "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
#Seasonality of transmission.

G=nx.from_pandas_edgelist(zoek('winter').append(zoek('seasonality').append(zoek('summer')).append(zoek('November')) ) , "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
#Prevalence of asymptomatic shedding and transmission (e.g., particularly children).

G=nx.from_pandas_edgelist(zoek('children').append( zoek('prevalence') ), "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()

#Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).



G=nx.from_pandas_edgelist(zoek('surface').append( zoek('') ), "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()

G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="indicated"], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="identified"], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="investigated"], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="showed"], "source", "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes

nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
# What is known about transmission, incubation, and environmental stability?

# increased, decreased, inhibited,inhibits,affects,results

kg_df[kg_df.edge=='decreased']