# This program creates a search engine and displays sample results for 3 queries. It searches the CORD-19 Database

#Output format:

# (Query Number) (Rank of Article for that query) (Article ID)

import os

import re

import json

import csv

import pickle as pkl

import tarfile

import numpy as np

import pandas as pd

import urllib.request

from langdetect import detect

from dataclasses import dataclass

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import *

from rank_bm25 import BM25Okapi



porter = PorterStemmer()



# Dataclass Article

@dataclass

class Article:

    ident      : str    # paper_id

    title      : str

    abstract   : str



# Download and unpack the collection

def getData():

    urls = ['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/comm_use_subset.tar.gz', 

            'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/noncomm_use_subset.tar.gz', 

            'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/custom_license.tar.gz', 

            'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz']



    # Create data directory

    try:

        os.mkdir('./data')

        print('Directory created')

    except FileExistsError:

        print('Directory already exists')



    # Download all files

    for i in range(len(urls)):

        print(i)

        urllib.request.urlretrieve(urls[i], './data/file'+str(i)+'.tar.gz')

        print('Downloaded file '+str(i+1)+'/'+str(len(urls)))

        tar = tarfile.open('./data/file'+str(i)+'.tar.gz')

        tar.extractall('./data')

        tar.close()

        print('Extracted file '+str(i+1)+'/'+str(len(urls)))

        os.remove('./data/file'+str(i)+'.tar.gz')



# Remove punctuation -> returns a string of words

def removePunc(sentence : str):

    words = ''

    if len(sentence) > 0:

        words = word_tokenize(sentence)

    return words



# Check if given word is a stopword -> returns a boolean

def isStopword(word : str):

    official_stopwords = np.array(stopwords.words('english'))

    more_stopwords = np.array(["can't", 'cannot', 'could', "he'd", "he'll", 

      'hi', "i'd", "i'll", "i'm", "i've", "let's", 'ought', 

      "she'd", "she'll", "that's", "there's", "they'd", "they'll", 

      "they're", "they've", "we'd", "we'll", "we're", "we've", 

      "what's", "when's", "where's", "who's", "why's", 'would', 'follow'])

    return (word.lower() in official_stopwords or word.lower() in more_stopwords)



# Stem words -> returns a string of stemmed words

def stemSentence(sentence : list):

    f = lambda w: porter.stem(w)

    stemmed = list(map(f, sentence))

    return ' '.join(stemmed)



# Parse given sentence -> returns a string of parsed text

def parse(text : str):

    if len(text) > 0:

        punc_removed = removePunc(text)

        stops_removed = np.array([word for word in punc_removed if not isStopword(word)])

        stemmed = stemSentence(stops_removed).lower()

    else:

        stemmed = ''

    return stemmed

    

# Get text of given category -> returns a string of text

def getText(category : dict):

    output = ""

    for x in range(len(category)):

        output += category[x]['text']

        output += " "

    return output[:-1]



# Create an Article from given json file -> returns an Article

def createArticle(doc, ident):

    realtitle = doc['metadata']['title']

    title = parse(realtitle)

    abstract = parse(getText(doc['abstract']))

    atcl = Article(ident, title, abstract)

    return atcl



# Check if title and/or abstract is an empty string -> returns a boolean

def exclude(title, abstract):

    if title == "" and len(abstract) == 0:

        return True

    elif title != "":

        try:

            if detect(title) != 'en':

                return True

        except Exception:

            return False

    elif len(abstract) > 0:

        try:

            if detect(abstract[0]['text'] != 'en'):

                return True

        except Exception:

            return False

    else:

        return False



# Iterate through the collection and extract key information from each article (Task 1)

def extract():

    extract_csv = csv.writer(open("extract.csv", "w"))

    extract_csv.writerow(['ID', 'titles', 'abstracts'])



    # Iterate through all files in the data directory

    for subdir, dirs, files in os.walk('./data'):

        for file in files:

            if file == ".DS_Store":

                continue

            with open(os.path.join(subdir, file)) as f:

                doc = json.load(f)

                title = doc['metadata']['title']

                abstract = doc['abstract']

                if exclude(title, abstract):

                    continue

                ident = doc['paper_id']

                new_article = createArticle(doc, ident)

                extract_csv.writerow([ident, new_article.title, new_article.abstract])



    # print("Extraction complete")



# Combine weighted repeats of title and abstract into one string -> returns an np array

def combineTextWeight(titles, abstracts):

    everything = []

    for i in range(len(titles)):

        t = ' '.join([titles[i]] * 3)

        a = abstracts[i]

        item = (t + ' ' + a).split(" ")

        everything.append(item)

    return np.array(everything)



# Organize the collection (Task 2)

def organize():

    df = pd.read_csv('extract.csv').fillna(' ')

    

    all_abstracts = []

    with open('extract.csv', 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter=',')

        next(csvfile)

        for rows in reader:

            all_abstracts.append(rows[2])



    titles1 = ' '.join(df.titles).split(' ')

    abstracts1 = ' '.join(all_abstracts).split(' ')

    distinct_words = list(set(titles1 + abstracts1))



    all_words = combineTextWeight(df.titles,all_abstracts)



    bm25 = BM25Okapi(all_words)

    bm25_pkl = open('bm25.pickle', "wb")

    pkl.dump(bm25, bm25_pkl, protocol=4)

    bm25_pkl.close()



    # print("Organization complete")



def retrieve(queries):

    bm25 = pkl.load(open("bm25.pickle", "rb"))

    paper_ids = list(pd.read_csv('extract.csv').fillna(' ').ID)

    titles = list(pd.read_csv('extract.csv').fillna(' ').titles)



    results = []

    processed_queries = [parse(query) for query in queries]



    for quer in processed_queries:

        each_results = []

        tokenized_quer = quer.split(' ')

        scores = bm25.get_scores(tokenized_quer)



        # Sort the scores to get the indices of the top 100 scores

        ind = np.argpartition(scores, -100)[-100:]



        # sortedInd holds the top indices in sorted descending order

        sortedInd = reversed(ind[np.argsort(scores[ind])])

        for index in sortedInd:

            each_results.append(paper_ids[index])

        results.append(each_results)



    # Output results

    for query in range(len(results)):

        for rank in range(len(results[query])):

            print(str(query+1)+'\t'+str(rank+1)+'\t'+str(results[query][rank]))



    # print("Retrieval complete")



def main():

    getData()

    extract()

    organize()

    q = ['coronavirus origin', 'coronavirus response to weather changes', 'coronavirus immunity']

    retrieve(q)

    

if __name__ == "__main__":

    main()