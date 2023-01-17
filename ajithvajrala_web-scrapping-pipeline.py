#import libraries

import urllib.request

from bs4 import BeautifulSoup, SoupStrainer

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

import nltk

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.tag import pos_tag

import requests

import json

import re

from nltk.probability import FreqDist

from nltk.corpus import stopwords

from nltk import conlltags2tree, tree2conlltags

from nltk import word_tokenize, pos_tag, ne_chunk

from nltk.tree import Tree

from nltk.tokenize.toktok import ToktokTokenizer

import math

from textblob import TextBlob as tb

import httplib2

from nltk.stem.porter import PorterStemmer

from nltk.stem import LancasterStemmer

from nltk.stem import WordNetLemmatizer

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import linear_kernel

import spacy

from spacy import displacy

nlp = spacy.load('en_core_web_sm')

import warnings

warnings.filterwarnings('ignore')
#______________links in Page, code changes starts____________________# 

def get_links(url):

    #read the url

    resp = urllib.request.urlopen(url)

    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))

    links =[]

    for link in soup.find_all('a', href=True):

        links.append(link['href'])

    #return all links present in that url

    return links

#______________links in Page, code changes End____________________# 





#_____________Get Meta Data, code changes start______________________#

def get_metadata(url):

    response = requests.get(url)

    soup = BeautifulSoup(response.text)

    metas = soup.find_all('meta')

    meta_data =  [ meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'description' ]

    meta_data = [w.rstrip() for w in meta_data]

    meta_data = [w.replace('\n',' ') for w in meta_data] 

    return meta_data

#_______________Get Meta Data, code changes end_____________________#







#_____________HTML Parsing, code changes starts____________________# 

def url_to_string(url):

    res = requests.get(url)

    html = res.text

    soup = BeautifulSoup(html, 'html5lib')

    for script in soup(["script", "style", 'aside']):

        script.extract()

    data = " ".join(re.split(r'[\n\t]+', soup.get_text()))

    return data

#_____________HTML Parsing, code changes END____________________# 





#_____________Extracting nouns, noun forms in document, code changes start__________#

def extract_noun_forms(data): 

    nouns_list = []

    sentences = sent_tokenize(data)

    for sent in sentences:

        #' '.join((e for e in sent if e.isalnum())

        tokens = nltk.word_tokenize(sent)

        tagged = nltk.pos_tag(tokens)    #Part-of-Speech Tagging'

        nouns = [word for word, pos in tagged if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]

        nouns_list.append(nouns)

    nouns_list = [x for sublist in nouns_list for x in sublist]

    return nouns_list

#_____________Extracting nouns in document, code changes END__________#





#_____________Entity and Relation Extraction, code changes starts____________________#

# i-o-b tagging, places, organisations, Names, Landmarks tagging

def get_continuous_chunks(text):

    chunked = ne_chunk(pos_tag(word_tokenize(text)))

    continuous_chunk = []

    current_chunk = []

    for i in chunked:

        if type(i) == Tree:

            current_chunk.append(" ".join([token for token, pos in i.leaves()]))

        elif current_chunk:

            named_entity = " ".join(current_chunk)

            if named_entity not in continuous_chunk:

                continuous_chunk.append(named_entity)

                current_chunk = []

        else:

            continue

    return continuous_chunk

#_____________Entity and Relation Extraction, code changes End____________________#





#_____________NAMED Entity Recognition and Extraction using spacy, code changes End____________________#

def generate_ner_tags(doc):

    out_list =[]

    for word in doc:

        #entities[word] = str(i.ent_iob_) + "-"+ str(i.ent_type_)

        if word.ent_type_ in ["ORG", "EVENT", "PERSON", "NORP", "GPE", "MONEY", "LOC", "WORK_OF_ART"]:

            my_list = [word, str(word.ent_iob_) + "-" + str(word.ent_type_)]

            out_list.append(my_list)            

    return out_list

#_____________NAMED Entity Recognition and Extraction using spacy, code changes End____________________#







#________________Text pre-processing code changes starts_______________#

stop_words=set(stopwords.words("english"))  

tokenizer = ToktokTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')

pattern = r'[^a-zA-z0-9\s]' #include digits  #r'[^a-zA-z\s]' remove digits also

porter = PorterStemmer()

lancaster = LancasterStemmer()



#Pre-processing: Sentence Splitting, Tokenization and Normalization 

def pre_process(text):

    text = re.sub(pattern, '', text)

    tokens = tokenizer.tokenize(text)

    tokens = [token.strip() for token in tokens]

    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopword_list]

    filtered_tokens = [lancaster.stem(word) for word in filtered_tokens]

    filtered_text = ' '.join(filtered_tokens)    

    return filtered_text

#_______________Text pre-processing code changes end_____________#





#______________Saving Files, code changes start_____________#

def save_text_file(filename, data):

    with open(filename, 'w') as f:

        for item in data:

            f.write("%s\n" % item)

            

def save_json(filename, data):

    with open(filename, 'w') as fp:

        json.dump(data, fp)

        

def save_html(file_name,html):

    with open(file_name, 'w', encoding='utf-8') as f:

         f.write(html)

#_____________Saving File, code changes End_____________#







#____________TF-IDF Updated Code changes start___________________#

def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)



def extract_topn_from_vector(feature_names, sorted_items, topn):

    #get the feature names and tf-idf score of top n items

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



def get_tf_idf(doc, corpus, num):

    cv=CountVectorizer(stop_words=stop_words)

    word_count_vector=cv.fit_transform(corpus)

    

    #TfidfTransformer to Compute Inverse Document Frequency (IDF)

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

    tfidf_transformer.fit(word_count_vector)

    

    feature_names= cv.get_feature_names()

    #generate tf-idf for the given document

    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

     #sort the tf-idf vectors by descending order of scores

    sorted_items=sort_coo(tf_idf_vector.tocoo())

     #extract only the top n;

    keywords=extract_topn_from_vector(feature_names,sorted_items,num)

    return keywords

#____________________TF IDF updated Code changes end_________________#







#____________Web search using cosine similarity, Code changes start_______________#

def web_search(text):

    cv=CountVectorizer( stop_words=stop_words)

    word_count_vector=cv.fit_transform(corpus)

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

    tfidf_transformer.fit(word_count_vector)

    full_x = tfidf_transformer.transform(cv.transform(corpus))

    sample1 = tfidf_transformer.transform(cv.transform([pre_process(text)]))

    cosine_similarities = linear_kernel(sample1,full_x).flatten()

    #return the first link that closely matches the text entered and the corresponding links metadata 

    url = urls[cosine_similarities.argsort()[:-5:-1][0]] 

    return  url, get_metadata(url)

#____________Web search using cosine similarity, Code changes End_______________#
url1 = "https://csee.essex.ac.uk/staff/udo/index.html"

url2 = "https://www.essex.ac.uk/departments/computer-science-and-electronic-engineering"



urls = [url1, url2]



data = []

#HTML Parsing and saving all the text in data

for i, url in enumerate(urls):

    filename ="html_parsed_doc_" + str(i+1)

    text = url_to_string(url)

    save_text_file(filename, sent_tokenize(text))

    data.append(text)
#saving meta data present in each link

for i, dat in enumerate(urls):

    filename ="Meta_data_in_link_" + str(i+1)

    text = get_metadata(url)

    save_text_file(filename, text)
#saving all the links present in each document

for i,url in enumerate(urls):

    filename ="links_in_doc_" + str(i+1)

    text = get_links(url)

    save_text_file(filename, text)
#saving all the Named Entities present in each document

for i,dat in enumerate(data):

    filename ="Named_Entities_in_doc_" + str(i+1)

    text = get_continuous_chunks(dat)

    save_text_file(filename, text)
#NER recognition using spacy and saving importnat tags like PERSON, LOR, ORG, 

for i, dat in enumerate(data):

    doc = nlp(data[i])

    entities = generate_ner_tags(doc)

    filename = "NER_in_doc_" + str(i+1)

    save_text_file(filename, entities)
#saving all the Nouns and noun forms present in each document

for i, dat in enumerate(data):

    filename ="Noun_and_noun_forms_in_doc_" + str(i+1)

    text = extract_noun_forms(dat)

    save_text_file(filename, text)
#Extracting corpus for calculating tf-idf scores

corpus =[]

for dat in data:

    corpus.append(pre_process(dat))

    

#saving all the TF-IDF scores in each document

for i, dat in enumerate(corpus):

    keywords = get_tf_idf(dat, corpus, num=20)

    filename = "top_tf_idf_scores_in_doc_" + str(i+1) + ".json"

    save_json(filename, keywords)  
sample = "Udo"

web_search(sample)
sample = "robotics"

web_search(sample)
#saving NER visually as html files to view whole data in page and corresponding tags

for i, dat in enumerate(data):

    doc = nlp(dat)

    html = displacy.render(doc, style='ent', page=True)

    filename = "NER_Image_in_doc_" + str(i+1) + ".html"

    save_html(filename, html)