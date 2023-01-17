import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
import nltk

from nltk.corpus import stopwords

import re

import string

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

stopword = set(stopwords.words("english"))
text = 'Together, with its affiliates, agents, employees, subsidiaries and representatives, hereinafter collectively referred to as "Disclosing Party" and Google Securities (USA) LLC ("Receiving Party"), each a Party and, together, hereinafter collectively referred to as the "Parties") have expressed their interest in discussing whether there is a basis for the Parties to consider a possible transaction for two specialty facilities located in Mesa and Sun City, Arizona. As a condition to being furnished Information (as defined below), Receiving Party agrees to treat all Information of Disclosing Party in accordance with the provisions of this letter agreement ("Agreement") and to take or abstain from taking certain other actions set forth in this Agreement (it being understood that Receiving Party its affiliates and the transaction contemplated herein and other such information (including such information communicated orally or contained on any computer tapes, computer disks or any other form of electronic or magnetic media). To Receiving Party or any of its Representatives, together with all analyses, compilations, studies or other documents, records or data (including information contained on any computer tapes, computer disks or any other form of electronic or magnetic media) prepared by Receiving Party and/or its Representatives that contain or otherwise reflect or are generated from such documents and information. This document expires on Friday, 28-June-2019.'
#print(text)
### 1. Remove Punctuation

#Function to Remove Punctuation

def remove_punct(text):

    text_no_punct = "".join([char for char in text if char not in string.punctuation])

    return text_no_punct
### 2. Tokenization

#Function to Tokenize words

def tokenize(text):

    #tokens = sent_tokenize(text)

    tokens = word_tokenize(text)

    return tokens
### 3. Remove Stopwords

#Function to Remove Stopwords

def remove_stopwords(text):

    stopword = set(stopwords.words("english"))

    text = text.lower()

    text = remove_punct(text)

    tokens = tokenize(text)

    tokens = [token.strip() for token in tokens]



    text = [word for word in tokens if word not in stopword]

    text = " ".join(text) 

    return text
### 4. Preprocessing Data: Using Stemming

def stemming(tokenized_text):

    ps = PorterStemmer()

    tokenized_text = tokenize(tokenized_text)

    text = [ps.stem(word) for word in tokenized_text]

    return text
def lemmatize_text(tokenized_text):  

    lemmatizer = WordNetLemmatizer()

    

    stopword = set(stopwords.words("english"))

    tokenized_text = remove_punct(tokenized_text)

    tokens = tokenize(tokenized_text)

    meaningful_words = [w for w in tokens if not w in stopword]

    text = [lemmatizer.lemmatize(word) for word in meaningful_words]

    return text



print(lemmatize_text(text))
def clean_text(raw_text):

    corpus = []

    text = remove_stopwords(raw_text)

    #meaningful_words = stemming(text)

    meaningful_words = lemmatize_text(text)

    clean_text = " ".join( meaningful_words )

    corpus.append(clean_text)

    return corpus
clean_text = clean_text(text)

print(clean_text)
### 7. Preprocessing Data: POS Tagging

def pos_tagging(text):    

    #text = nltk.word_tokenize(text)

    text = nltk.pos_tag(text.split())

    return text



pos_tag = pos_tagging(text)
print(pos_tag)

#pd.DataFrame(pos_tag, columns=['Word', 'POS tag'])
#Chunking with Regular Expressions

grammar = r"""

  Chunk: {<DT>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun {<DT|PP\$>?<JJ>*<NN>}

         {<NNP>+}           # chunk sequences of proper nouns

"""

cp = nltk.RegexpParser(grammar)

cs = cp.parse(pos_tag)

#print(cs)
#Chinking: to remove anything from text

grammar = r"""

  Chunk: {<.*>+}              # Chunk everything

         }<VB.?|IN|DT>+{      # Chink sequences of VBD and IN

"""

cp = nltk.RegexpParser(grammar)

chink = cp.parse(pos_tag)

#print(chink)
from nltk.chunk import conlltags2tree, tree2conlltags

from pprint import pprint

iob_tagged = tree2conlltags(cs)

#pprint(iob_tagged)

#pd.DataFrame(iob_tagged, columns=['Word', 'Chunk tag', 'Type'])
namedEnt = nltk.ne_chunk(pos_tag)

#ne_tree = ne_chunk(pos_tag(word_tokenize(text)))

#print(namedEnt)
#Entity

import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()



doc = nlp(text)

pprint([(X.text, X.label_) for X in doc.ents])
#"B" means the token begins an entity, "I" means it is inside an entity, "O" means it is outside an entity, and "" means no entity tag is set

pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
labels = [x.label_ for x in doc.ents]

Counter(labels)
df = pd.DataFrame(clean_text, columns=['clean_text'])
from sklearn.feature_extraction.text import CountVectorizer

import re

cv = CountVectorizer(stop_words=stopword, max_features=10000, ngram_range=(1,1))

X = cv.fit_transform(df["clean_text"])

#print(X)

#print(cv.get_feature_names())



# Show the Bag-of-Words Model as a pandas DataFrame

feature_names = cv.get_feature_names()

pd.DataFrame(X.toarray(), columns = feature_names)
#from sklearn.feature_extraction.text import TfidfTransformer

 

#tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)

#tfidf_transformer.fit(X)

# get feature names

#feature_names = cv.get_feature_names()

 

# fetch document for which keywords needs to be extracted

#doc = clean_text[0]

 

#generate tf-idf for the given document

#tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vectorizer = TfidfVectorizer()

values = tfidf_vectorizer.fit_transform(df["clean_text"])



# Show the Model as a pandas DataFrame

feature_names = tfidf_vectorizer.get_feature_names()

pd.DataFrame(values.toarray(), columns = feature_names)
#Function for sorting tf_idf in descending order

from scipy.sparse import coo_matrix

def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):

    """get the feature names and tf-idf score of top n items"""

    

    #use only topn items from vector

    sorted_items = sorted_items[:topn]

 

    score_vals = []

    feature_vals = []

    

    # word index and corresponding tf-idf score

    for idx, score in sorted_items:

        

        #keep track of feature name and its corresponding score

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])

 

    #create a tuples of feature,score

    #results = zip(feature_vals,score_vals)

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    

    return results





#sort the tf-idf vectors by descending order of scores

sorted_items=sort_coo(values.tocoo())

#extract only the top n; n here is 10

keywords=extract_topn_from_vector(feature_names,sorted_items,5)

 

# now print the results

print("\nAbstract:")

print(df["clean_text"])

print("\nKeywords:")

for k in keywords:

    print(k,keywords[k])
sentences = [x for x in doc.sents]

print(sentences[:])
[(x.orth_,x.pos_, x.lemma_) for x in [y 

                                      for y

                                      in nlp(str(sentences[:])) 

                                      if not y.is_stop and y.pos_ != 'PUNCT']]
dict([(str(x), x.label_) for x in nlp(str(sentences[:])).ents])
displacy.render(doc, jupyter=True, style='ent')