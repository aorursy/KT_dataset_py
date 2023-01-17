# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#libraries for getting data and extracting

import os

import urllib.request

import tarfile

import json

import pandas as pd

import numpy as np

from tqdm import tqdm





#libraries for text preprocessing

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

nltk.download('wordnet') 

from nltk.stem.wordnet import WordNetLemmatizer



#libraries for keyword extraction with tf-idf

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import coo_matrix



#libraries for reading and writing files

import pickle



#libraries for BM25

!pip install rank_bm25

from rank_bm25 import BM25Okapi
def getData():

    urls = ['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/comm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/noncomm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/custom_license.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz']



    # Create data directory

    try:

        os.mkdir('./data')

        print('Directory created')

    except FileExistsError:

        print('Directory already exists')



    #Download all files

    for i in range(len(urls)):

        urllib.request.urlretrieve(urls[i], './data/file'+str(i)+'.tar.gz')

        print('Downloaded file '+str(i+1)+'/'+str(len(urls)))

        tar = tarfile.open('./data/file'+str(i)+'.tar.gz')

        tar.extractall('./data')

        tar.close()

        print('Extracted file '+str(i+1)+'/'+str(len(urls)))

        os.remove('./data/file'+str(i)+'.tar.gz')
def preprocess(text):

    #define stopwords

    stop_words = set(stopwords.words("english"))

    #Remove punctuations

    text = re.sub('[^a-zA-Z]', ' ', text)

    #Convert to lowercase

    text = text.lower()

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    # remove special characters and digits

    text=re.sub("(\\d|\\W)+"," ",text)

    ##Convert to list from string

    text = text.split()

    ##Stemming

    ps=PorterStemmer()

    text = [ps.stem(word) for word in text if not word in stop_words]

    #Lemmatisation

    lem = WordNetLemmatizer()

    text = [lem.lemmatize(word) for word in text if not word in  stop_words] 

    text = " ".join(text) 

    

    return text
def extract():

    #create our collection locally in the data folder

    

    #creating our initial datastructure

    x = {'paper_id':[], 'title':[], 'abstract': []}

    

    #Iterate through all files in the data directory

    for subdir, dirs, files in os.walk('./data'):

        for file in tqdm(files):

            with open(os.path.join(subdir, file)) as f:

                data = json.load(f)

                

               #Append paper ID to list

                x['paper_id'].append(data['paper_id'])

               #Append article title to list & preprocess the text

                x['title'].append((data['metadata']['title']))

                

                #Append abstract text content values only to abstract list & preprocess the text

                abstract = ""

                for paragraph in data['abstract']:

                    abstract += paragraph['text']

                    abstract += '\n'

                #if json file no abstract in file, set the body text as the abstract (happens rarely, but often enough that this edge case matters)

                if abstract == "": 

                    for paragraph in data['body_text']:

                        abstract += paragraph['text']

                        abstract += '\n'

                x['abstract'].append(preprocess(abstract))

                

    #Create Pandas dataframe & write to pickle file

    df = pd.DataFrame.from_dict(x, orient='index')

    df = df.transpose()

    pickle.dump( df, open( "full_data_processed_FINAL.p", "wb" ) )

    return df
def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
def extract_topn_from_vector(feature_names, sorted_items, topN):

    #use only topn items from vector

    sorted_items = sorted_items[:topN]

 

    score_vals = []

    feature_vals = []

    # word index and corresponding tf-idf score

    for idx, score in sorted_items:

        #keep track of feature name and its corresponding score

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])

 

    #create a tuples of feature,score

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    return results
def getAbstractKeywords(entry, cv, X, tfidf_transformer, feature_names, topN):

    abstract = entry['abstract']

    

    #first check that abstract is full

    if type(abstract) == float:

        return []

 

    #generate tf-idf for the given document

    tf_idf_vector=tfidf_transformer.transform(cv.transform([abstract])) 

    #sort the tf-idf vectors by descending order of scores

    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the topN # items

    keywords_dict=extract_topn_from_vector(feature_names,sorted_items,topN)

    #just want words themselves, so only need keys of the dictionary

    keywords = list(keywords_dict.keys()) 

     

    return keywords
def getTitleKeywords(entry):

    title = entry['title']  

    title = preprocess(title)

    #first check that the title of that entry is full

    if type(title) == float:

        return []

    

    keywords_title = title.split(' ')

    return keywords_title
def getFinalKeywords(entry, cv, X, tfidf_trans, feature_names, topN):

    #get keywords from abstract and title

    fromAbstract = getAbstractKeywords(entry, cv, X, tfidf_trans, feature_names, topN)

    fromTitle = getTitleKeywords(entry)

    #concatenate two lists

    finalKeywords = fromAbstract + fromTitle

    #convert to set and then back to list to ensure there are no duplicates in list

    final_no_duplicates = list(set(finalKeywords))

    return final_no_duplicates
def getCorpus(articlesDf):

    #creating a new dataframe, abstractDf, of just the abstracts, so that we don't modify the original dataframe, articlesDf

    abstractDf = pd.DataFrame(columns = ['abstract'])

    #filling abstractDf with the abstract column from articlesDf

    abstractDf['abstract'] = articlesDf['abstract']

    #converting column of dataframe to a list

    corpus = abstractDf['abstract'].to_list()

    return corpus
def addKeywords(df, topN, makeFile, fileName):

    #defining stopwords

    stop_words = set(stopwords.words("english"))



    #creating following variables that are needed for keyword extract from abstract, using tf-idf methodology,

    #all input in getFinalKewords method

    corpus = getCorpus(df)

    cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=1000, ngram_range=(1,1))    

    X=cv.fit_transform(corpus)

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

    tfidf_transformer.fit(X)

    feature_names=cv.get_feature_names()

    

    #adding keywords article to dataframe

    df = df.reindex(columns = ['paper_id', 'title', 'abstract','keywords'])                

    #getting keywords for each entry in article dataframe -- using apply to be more efficient

    df['keywords'] = df.apply(lambda row: getFinalKeywords(row, cv, X, tfidf_transformer, feature_names, topN), axis=1)



    #make pickle file depending on user input

    if makeFile == True:

        pickle.dump( df, open( fileName, "wb" ) )

    return df  
def createInvertedIndices(df):

    numEntries = df.shape[0]

    invertInd = {}

    

    for i in range (numEntries):

        entry = df.iloc[i]

        paper_id = entry['paper_id']    

        keywords = entry['keywords']

        for k in keywords:

            if k not in invertInd:

                invertInd[k] = []

                invertInd[k].append(paper_id)

            else:

                invertInd[k].append(paper_id)

    return invertInd
def organize():

    df_without_keywords = pickle.load(open("full_data_processed_FINAL.p", "rb"))

    df_with_keywords = addKeywords(df_without_keywords, 10, False, "full_data_withKeywords_FINAL.p")

    invertedIndices = createInvertedIndices(df_with_keywords)

    pickle.dump( invertedIndices, open( "invertedIndices_FINAL.p", "wb" ) )
def getPotentialArticleSubset(query):

    #load in inverted indices

    invertedIndices = pickle.load(open("invertedIndices_FINAL.p", "rb"))

    

    #preprocess query and split into individual terms

    query = preprocess(query)

    queryTerms = query.split(' ')

    

    potentialArticles = []

    #concatenate list of potential articles by looping through potential articles for each word in query

    for word in queryTerms:

        if word in invertedIndices: #so if someone types in nonsensical query term that's not in invertedIndices, still won't break!

            someArticles = invertedIndices[word]

            potentialArticles = potentialArticles + someArticles

            

    #convert to set then back to list so there are no repeat articles

    potentialArticles = list(set(potentialArticles))

    return potentialArticles
def bm25(articles, df_dic, title_w, abstract_w, query):

    corpus_title = []

    corpus_abstract = []

    

    for article in articles:

        arr = df_dic.get(article)

        #title

        if type(arr[0]) != float:

            preprocessedTitle = preprocess(arr[0])

            corpus_title.append(preprocessedTitle)

        else:

            corpus_title.append(" ")

        

        #abstract

        if type(arr[1]) != float:

            preprocessedAbst = preprocess(arr[1])

            corpus_abstract.append(preprocessedAbst)

        else:

            corpus_abstract.append(" ")

            

    query = preprocess(query)

    

    tokenized_query = query.split(" ")

    

    tokenized_corpus_title = [doc.split(" ") for doc in corpus_title]

    tokenized_corpus_abstract = [doc.split(" ") for doc in corpus_abstract]

    

    #running bm25 on titles

    bm25_title = BM25Okapi(tokenized_corpus_title)

    doc_scores_titles = bm25_title.get_scores(tokenized_query)

    #weighting array

    doc_scores_titles = np.array(doc_scores_titles)

    doc_scores_titles = doc_scores_titles**title_w

    

    #running bm25 on abstracts

    bm25_abstract = BM25Okapi(tokenized_corpus_abstract)

    doc_scores_abstracts = bm25_abstract.get_scores(tokenized_query)

    #weighting

    doc_scores_abstracts = np.array(doc_scores_abstracts)

    doc_scores_abstracts = doc_scores_abstracts ** abstract_w

    

    #summing up the two different scores

    doc_scores = np.add(doc_scores_abstracts,doc_scores_titles)

    

    #creating a dictionary with the scores

    score_dict = dict(zip(articles, doc_scores))

    

    #creating list of ranked documents high to low

    doc_ranking = sorted(score_dict, key=score_dict.get, reverse = True)

    

    #get top 100

    doc_ranking = doc_ranking[0:100]

    

    for i in range(len(doc_ranking)):

        dic_entry = df_dic.get(doc_ranking[i])

        doc_ranking[i] = dic_entry[0]

    

    return doc_ranking
def retrieve(queries):

    #performing information retrieval

    df_without_keywords = pickle.load(open("full_data_processed_FINAL.p", "rb"))

    df_dic = df_without_keywords.set_index('paper_id').T.to_dict('list')

    results = []

    for q in queries:

        articles = getPotentialArticleSubset(q)

        result = bm25(articles,df_dic,1,2,q)

        results.append(result)



    #Output results

    for query in range(len(results)):

        for rank in range(len(results[query])):

            print(str(query+1)+'\t'+str(rank+1)+'\t'+str(results[query][rank]))

            
getData()

extract()

organize()

q = ['coronavirus origin',

'coronavirus response to weather changes',

'coronavirus immunity']

retrieve(q)