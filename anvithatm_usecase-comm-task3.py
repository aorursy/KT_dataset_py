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
#Read and other operations

import pandas as pd

import numpy as np



#Data cleaning

import string

import itertools 

import re

from nltk.stem import WordNetLemmatizer

from string import punctuation

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer



#tokenization

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize



#cosine similarity

from sklearn.metrics.pairwise import cosine_similarity



#Get nearest match word in a list for a given word

import difflib



#tfidf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



import csv



#convert string to list

import ast
level1_sub_questions = ["1. Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.",

                   "2. Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.",

                   "3. Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.",

                   "4. Evidence of whether farmers are infected, and whether farmers could have played a role in the origin",

                   "5. Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.",

                   "6. Experimental infections to test host range for this pathogen.",

                   "7. Animal host(s) and any evidence of continued spill-over to humans",

                   "8. Socioeconomic and behavioral risk factors for this spill-over",

                   "9. Sustainable risk reduction strategies"]
sub_categories_keywords = [["real-time tracking", "rapid dissemination","diagnostics and therapeutics", "track variations"],

                   ["geographic distribution", "genomic difference", "multi-lateral agreements", "nagoya protocol"],

                   ["livestock", "field surveillance", "genetic sequencing", "receptor binding", "reservoir"],

                   ["farmers"],

                   ["wildlife-livestock", "southeast asia"],

                   ["experimental infection"],

                   ["animal host"],

                   ["socio economic factors", "behavioural factors"],

                 ["risk reduction strategies"]]
my_stopwords = stopwords.words('english')



def cleanData(text, lower_case = False,custom_replace = False,remove_punc = False,remove_stops = False, custom_stop=False,stemming = False, lemmatization = False,remove_numeric=False):



    custom_stopwords = []

    custom_replace = ["Abstract","CC-BY-NC-ND 4.0","CC-BY-ND 4.0","International license is made available under a","author/funder, who has granted medRxiv a license to display the preprint in perpetuity","is the (which was not peer-reviewed)","The copyright holder for this preprint"]

    txt = str(text)

    

    # Remove urls and emails

    txt = re.sub(r'\b[https|http]+:[\S]+', ' ', txt, flags=re.MULTILINE)

    

    #convert to lower case

    if lower_case:

        txt = txt.lower()

        

    #remove custom replace

    if custom_replace:

        for cr in custom_replace:

             txt = txt.replace(cr.lower(),"")

    # Remove punctuation from text

    if remove_punc:

        txt = ''.join([c for c in text if c not in punctuation])



    if stemming:

        st = PorterStemmer()

        txt = " ".join([st.stem(w) for w in txt.split()])

    

    if lemmatization:

        wordnet_lemmatizer = WordNetLemmatizer()

        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])

    if remove_stops:

        txt = " ".join([w for w in txt.split() if w not in my_stopwords])

    if custom_stop:

        txt = " ".join([w for w in txt.split() if w not in custom_stopwords])

    if remove_numeric:

        txt = ''.join([i for i in txt if not i.isdigit()])



    return txt
csv_data_comm=pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")
csv_data_comm.shape
#for text

preprocessed_text_comm = csv_data_comm['text'].map(lambda x: cleanData(x,lower_case = True,custom_replace = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))



#for abstract

preprocessed_abs_comm = csv_data_comm['abstract'].map(lambda x: cleanData(x,lower_case = True,custom_replace = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))
tfidf_vect_comm_text = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1,1),max_features=8000)

tfidf_vect_comm_abs = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1,1),max_features=8000)



vect_scores_comm_text = tfidf_vect_comm_text.fit_transform(preprocessed_text_comm)

vect_scores_comm_abs = tfidf_vect_comm_abs.fit_transform(preprocessed_abs_comm)
import pickle

with open('article_comm_vectorizer_task3.pickle', 'wb') as fin:

    pickle.dump(tfidf_vect_comm_text, fin)

with open('abstract_comm_vectorizer_task3.pickle', 'wb') as fin:

    pickle.dump(tfidf_vect_comm_abs, fin)
feature_words = tfidf_vect_comm_text.get_feature_names()

tfidf_feature_scores = vect_scores_comm_text.toarray()
def get_word_imp_score_article(s_word,article_idx):

    search_word_repl = s_word.replace("-"," ")

    search_word_tokens = word_tokenize(search_word_repl)

    importance_score = 0

    for s_wd in search_word_tokens:

        if s_wd in feature_words:

            feature_index = feature_words.index(s_wd)

            #print(vect_scores.toarray()[i][feature_index])

            feature_score = tfidf_feature_scores[article_idx][feature_index]

            importance_score = importance_score + feature_score

        else:

            match_word = difflib.get_close_matches(s_wd, feature_words,cutoff=0.75)

            if match_word:

                feature_index = feature_words.index(match_word[0])

                #print(vect_scores.toarray()[i][feature_index])

                feature_score = tfidf_feature_scores[article_idx][feature_index]

                importance_score = importance_score + feature_score

    

    return importance_score
import time

def extract_article(search_words):

    final_articles = []

    final_articles_idx = []

    articles_scores = []

    

    for search_word in search_words:

        print("**search word**",search_word)

        articles_imp_score=[]

        search_word = search_word.lower()

        final_imp_score = 0

        for i in range(0,csv_data_comm.shape[0]):

            

            importance_score=0

            if search_word in csv_data_comm.iloc[i,5].lower():

                #print("inside if")

                importance_score = get_word_imp_score_article(search_word,i)

            else:

                #new_search_word = ""

                word_tokens = word_tokenize(search_word)

                #max_score = 0

                for word in word_tokens:

                    if word not in ["to","is","the","and","of","for"]:

                        match_word = difflib.get_close_matches(word, feature_words,cutoff=0.75)

                        if match_word:

                            new_search_word = match_word[0]

                            word_score = get_word_imp_score_article(new_search_word,i)

                            if word_score>importance_score:

                                importance_score = word_score



            articles_imp_score.append(importance_score)

            

        articles = [articles_imp_score.index(score) for score in articles_imp_score if score > 0.7*max(articles_imp_score)]

        

        final_articles_idx.append(articles)

       

    final_articles_idx = [j for i in final_articles_idx for j in i]

    return final_articles_idx
article_indexes = []



for kw in sub_categories_keywords:

    print(kw)

    art_idx = extract_article(kw)

    art_idx = list(set(art_idx))

    print("art_idx",art_idx)

    article_indexes.append(art_idx)

    #break

print("article_indexes ",article_indexes)

sub_task_number = list(range(1,len(article_indexes)+1))

print(sub_task_number)

art_idx_df = pd.DataFrame(list(zip(sub_task_number, article_indexes)), 

               columns =['Sub_task_number', 'Article Indexes'])

art_idx_df.to_csv("article_indexes_comm_task3.csv")