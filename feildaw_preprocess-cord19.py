# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    print(dirname)

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#additional imports

import glob

import json

import gensim

import nltk

import pickle #for writing to output

import gc
def readarticle(filepath):

    paperdata = {"paper_id" : None, "title" : None, "abstract" : None}

    with open(filepath) as file:

        filedata = json.load(file)

        paperdata["paper_id"] = filedata["paper_id"]

        paperdata["title"] = filedata["metadata"]["title"]

                

        if "abstract" in filedata:

            abstract = []

            for paragraph in filedata["abstract"]:

                abstract.append(paragraph["text"])

            abstract = '\n'.join(abstract)

            paperdata["abstract"] = abstract

            #print(abstract)

        else:

            paperdata["abstract"] = []

            #print("no abstract")



        #body_text = []

        #for paragraph in filedata["body_text"]:

        #    body_text.append(paragraph["text"])

        #body_text = '\n'.join(body_text)

        #paperdata["body_text"] = body_text

        

    return paperdata



def read_multiple(jsonfiles_pathnames):

    papers = {"paper_id" : [], "title" : [], "abstract" : []}

    for filepath in jsonfiles_pathnames:

        paperdata = readarticle(filepath)

        if len(paperdata["abstract"]) > 0: 

            papers["paper_id"].append(paperdata["paper_id"])

            papers["title"].append(paperdata["title"])

            papers["abstract"].append(paperdata["abstract"])

            #papers["body_text"].append(paperdata["body_text"])

            #print("not none")

        #else:

            #print("none")

    print(len(papers["paper_id"]))

    print(len(papers["title"]))

    print(len(papers["abstract"]))

    gc.collect()

    return papers
def make_bigram(tokenized_data, min_count = 5, threshold = 100):

    bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)

    #after Phrases a Phraser is faster to access

    bigram = gensim.models.phrases.Phraser(bigram_phrases)

    gc.collect()

    return bigram



def make_trigram(tokenized_data, min_count = 5, threshold = 100):

    bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)

    trigram_phrases = gensim.models.Phrases(bigram_phrases[tokenized_data], threshold = 100)

    #after Phrases a Phraser is faster to access

    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    gc.collect()

    return trigram

    
def readjson_retbodytext(jsonfiles_pathnames):

    print("reading json files")

    documents = read_multiple(jsonfiles_pathnames)

    print("writing documents dictionary to output for use in another kernel")

    with open("documents_dict.pkl", 'wb') as f:

        pickle.dump(documents, f)

    print("done writing documents dict.  Format is paper_id, title, body_text")

    gc.collect()

    return documents["abstract"]

    

def open_tokenize(jsonfiles_pathnames):

    

    body_text = readjson_retbodytext(jsonfiles_pathnames)

    

    print("removing stopwords, steming, and tokenizing.  This is expensive")

    tokenized_documents = gensim.parsing.preprocessing.preprocess_documents(body_text)

    print("done preprocessing documents. now writing to output to be used in another documents")

    with open("tokenized_documents.pkl", 'wb') as f:

        pickle.dump(tokenized_documents, f)

    print("done writing file")

    

    gc.collect()

    return tokenized_documents

    
metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

#metadata.head()

#print(metadata.shape)

metadata.info()
jsonfiles_pathnames = glob.glob("/kaggle/input/CORD-19-research-challenge/**/*.json", recursive = True)

print(len(jsonfiles_pathnames))


tokenized_documents = open_tokenize(jsonfiles_pathnames)
print("creating bigram model.  this is expensive, skipping trigrams")

bigram_model = make_bigram(tokenized_documents)

print("done creating bigram model. Writing model")

with open("bigram_model.pkl", 'wb') as f:

    pickle.dump(bigram_model, f)

print("done writing bigram model")