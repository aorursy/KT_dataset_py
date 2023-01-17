# Importing All The Needed Libraries

import numpy as np

import pandas as pd

import json

import math

import os

import string

import seaborn as sns

import re

import nltk

import gensim

from nltk import sent_tokenize, word_tokenize, PorterStemmer

from nltk.corpus import stopwords 
# Loading The Dataset 

articles = {}

for dirpath, subdirs, files in os.walk('/kaggle/input'):

    for x in files:

        if x.endswith(".json"):

            articles[x] = os.path.join(dirpath, x)        

data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
keywords = [("coronavirus", 10), ("covid-19", 10), ("corona", 10),("severity", 5), ("risk", 5), ("fatality", 5), ("symptomatic", 3)

            , ("groups", 3), ("risks", 5), ("vulnerability", 5), ("case-fatality", 5), ("mortality", 4), ("attack", 3), ("asymptomatic", 3)

            , ("hospitalized", 2), ("subclinical", 2), ("suspected", 2), ("hospitalised", 2), ("inpatients", 2), ("outpatients", 2)

            , ("children", 3), ("survivors", 2) , ("infants", 3), ("case-patients", 3), ("women", 3),("low-risk", 5), ("at-risk", 4)

            , ("child", 3), ("case-patient", 4), ("traveller", 3),("subgroups", 3), ("group", 3),("sexes", 3), ("genders", 3)

            , ("categories", 3), ("high-risk", 5)]
data = data[data['abstract'].notna()]
related_articles = []

for article in data.iterrows():

    score = 0

    for keyword in keywords:

        if(keyword[0] in article[1]['abstract'].lower()):

            score = score + keyword[1]

    if(score > 0):

        article[1]['score'] = score       

        related_articles.append(article[1])
related_articles.sort(key=lambda x: x['score'], reverse=True)
"""

 A function that returns the full_body from the dataset

   Parameters title: String, full_text_file: String

   Returns "body_text":  list of paragraphs in full body in the following format

                      [                  

                                            

                           {

                               "text": <str>,

                               "cite_spans": [],

                               "ref_spans": [],

                               "eq_spans": [],

                               "section": "Introduction"

                           },

                           ...

                           {

                               ...,

                               "section": "Conclusion"

                            }

                       ]

""" 





def get_full_text(title, full_text_file):

    

    # File is not available if full_text_file or title is NaN

    if(pd.isnull(title) or pd.isnull(full_text_file)):

        return "Full Text Is Not Accessible"



    # Set the path for the file

    dirname = '/kaggle/input/CORD-19-research-challenge'

    if(full_text_file == 'biorxiv_medrxiv'):

        dirname = dirname + '/biorxiv_medrxiv/biorxiv_medrxiv/'

    elif(full_text_file == 'comm_use_subset'):

        dirname = dirname + "/comm_use_subset/comm_use_subset/"

    elif(full_text_file == 'custom_license'):

        dirname = dirname + "/custom_license/custom_license/"

    elif(full_text_file == 'noncomm_use_subse'):

        dirname = dirname + "/noncomm_use_subset/noncomm_use_subset/"

    dirname = dirname + '/pdf_json'

    

    # Iterate over json files to find the paper with the specified title

    for _, _, filenames in os.walk(dirname):

        for filename in filenames:

            if filename.split('.')[-1] == 'json':

                path = os.path.join(dirname, filename)

                article = json.load(open(path, 'rb'))

                metadata = article['metadata']



                title = metadata['title'].strip()

                if(title ==  metadata['title']):

                    return article['body_text']
"""

A function to create the Frequency matrix of the words in each sentence.

"""

def create_frequency_matrix(sentences):

    frequency_matrix = {}

    stopWords = set(stopwords.words("english"))

    ps = PorterStemmer()



    for sent in sentences:

        freq_table = {}

        words = word_tokenize(sent)

        for word in words:

            word = word.lower()

            word = ps.stem(word)

            if word in stopWords:

                continue



            if word in freq_table:

                freq_table[word] += 1

            else:

                freq_table[word] = 1



        frequency_matrix[sent[:15]] = freq_table



    return frequency_matrix
"""

A function to calculate TermFrequency and generate a matrix

We’ll find the TermFrequency for each word in a paragraph.

Now, remember the definition of TF,

TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)

Here, the document is a paragraph, the term is a word in a paragraph.

"""

def create_tf_matrix(freq_matrix):

    tf_matrix = {}



    for sent, f_table in freq_matrix.items():

        tf_table = {}



        count_words_in_sentence = len(f_table)

        for word, count in f_table.items():

            tf_table[word] = count / count_words_in_sentence



        tf_matrix[sent] = tf_table



    return tf_matrix
"""

A fuction to creating a table for documents per words. Simplt

calculating “how many sentences contain a word".

"""

def create_documents_per_words(freq_matrix):

    word_per_doc_table = {}



    for sent, f_table in freq_matrix.items():

        for word, count in f_table.items():

            if word in word_per_doc_table:

                word_per_doc_table[word] += 1

            else:

                word_per_doc_table[word] = 1



    return word_per_doc_table
"""

A fuction to calculate IDF and generate a matrix

We’ll find the IDF for each word in a paragraph.

Now, remember the definition of IDF,

IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

Again the document here is a paragraph, and the term here is a word in a paragraph.

"""

def create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):

    idf_matrix = {}



    for sent, f_table in freq_matrix.items():

        idf_table = {}



        for word in f_table.keys():

            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))



        idf_matrix[sent] = idf_table



    return idf_matrix
"""

A function to calculate TF-IDF and generate a matrix

In simple terms, we are multiplying the values from both matrices "tf_matrix and idf_matrix" and generating new matrix.

"""

def create_tf_idf_matrix(tf_matrix, idf_matrix):

    tf_idf_matrix = {}



    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):



        tf_idf_table = {}



        for (word1, value1), (word2, value2) in zip(f_table1.items(),

                                                    f_table2.items()):  # here, keys are the same in both the table

            tf_idf_table[word1] = float(value1 * value2)



        tf_idf_matrix[sent1] = tf_idf_table



    return tf_idf_matrix
"""

A function to score a sentence by its word's TF by adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.

:rtype: dict

"""

def score_sentences(tf_idf_matrix) -> dict:

    



    sentenceValue = {}



    for sent, f_table in tf_idf_matrix.items():

        total_score_per_sentence = 0



        count_words_in_sentence = len(f_table)

        for word, score in f_table.items():

            total_score_per_sentence += score



        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence



    return sentenceValue
"""

A function to Find the average score from the sentence value dictionary

:rtype: int

"""

def find_average_score(sentenceValue) -> int:

    sumValues = 0

    for entry in sentenceValue:

        sumValues += sentenceValue[entry]



    # Average value of a sentence from original summary_text

    average = (sumValues / len(sentenceValue))



    return average
"A fucntion to generate the summary based on the given threshold. By comparing the sentence's score to the threshold"

def generate_summary(sentences, sentenceValue, threshold):

    sentence_count = 0

    summary = ''



    for sentence in sentences:

        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):

            summary += " " + sentence

            sentence_count += 1



    return summary
"""

A function to return the full text of the articles as list of tuples each tuple has the first element as the

section title and the second element is the all the paragraphs included in this sections in the full text.

"""

def article_sections(full_text):

    sections = []

    sections_with_text = []

    for paragraph in full_text:

        if(paragraph['section'] not in sections):

            sections.append(paragraph['section'])

            sections_with_text.append([paragraph['section'], paragraph['text']])

        else:

            for section_with_text in sections_with_text:

                if(section_with_text[0] == paragraph['section']):

                    section_with_text[1] = section_with_text[1] + paragraph['text']

                    break

    return sections_with_text

        
"""

A fuction that serves as a pipeline to summarize the given articles and retruns the summary as a dictionary with the title,

authors, publish time and a list of tuples for sections each tuple has the section title as the first element and the

summary of this setion as the second element.

"""

def summarize_article(article):

    summary = {}

    summary['title'] = article['title']

    summary['authors'] = article['authors']

    summary['publish_time'] = article['publish_time']

    summary['sections_summaries'] = []

    

    if(pd.isnull(article['full_text_file'])):

        summary['sections_summaries'] = "Full Text Is Not Acessible!"

        return summary

        

    full_text = get_full_text(article['title'], article['full_text_file'])

    sections = article_sections(full_text)

    

    for section in sections:

        # 1 Preprocessing Sections' Paraghraphs

        paraghraph_without_links = re.sub(r'^https?:\/\/.*[\r\n]*', '', section[1], flags=re.MULTILINE)

        # Removing doi urls from articles

        preprocessed_text  = re.sub(r'https://doi.org/10.1101/752592', '', paraghraph_without_links, flags=re.MULTILINE)

        preprocessed_text  = re.sub(r'https://doi.org/10.1101/2020.03.26.009993', '',  preprocessed_text, flags=re.MULTILINE)

        # 2 Sentence Tokenize

        sentences = sent_tokenize(preprocessed_text)

        total_documents = len(sentences)



        # 3 Create the Frequency matrix of the words in each sentence.

        freq_matrix = create_frequency_matrix(sentences)

        '''

        Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.

        '''

        # 4 Calculate TermFrequency and generate a matrix

        tf_matrix = create_tf_matrix(freq_matrix)



        # 5 creating table for documents per words

        count_doc_per_words = create_documents_per_words(freq_matrix)



        '''

        Inverse document frequency (IDF) is how unique or rare a word is.

        '''

        # 6 Calculate IDF and generate a matrix

        idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)



        # 7 Calculate TF-IDF and generate a matrix

        tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)



        # 8 Important Algorithm: score the sentences

        sentence_scores = score_sentences(tf_idf_matrix)



        # 9 Find the threshold

        threshold = find_average_score(sentence_scores)



        # 10 Important Algorithm: Generate the summary

        section_summary = generate_summary(sentences, sentence_scores, 0.75 * threshold)

        

        summary['sections_summaries'].append([section[0], section_summary])

        

    return summary
summarized_articles = []

for article in related_articles[:20]:

    summarized_articles.append(summarize_article(article))



for summary in summarized_articles:

    print("Title:")

    print(summary['title'])

    print("Publish Date:")

    print(summary['publish_time'])

    print("Summary:")

    if(summary['sections_summaries'] == "Full Text Is Not Acessible!"):

         print("Full Text Is Not Acessible!")

    else:

         for section in summary['sections_summaries']:

            print(section[0] + ":")

            print()

            print(section[1])

            print()

    print("-------------------------------------------------------------------------------------------")

    print('\n')

    
"""

A fuction to summarize the given articles  using gensim.summarize() retruns the summary as a dictionary with the title,

authors, publish time and a list of tuples for sections each tuple has the section title as the first element and the

summary of this setion as the second element.

"""

def summarize_gensim(article):

    summary = {}

    summary['title'] = article['title']

    summary['authors'] = article['authors']

    summary['publish_time'] = article['publish_time']

    summary['sections_summaries'] = []

    

    if(pd.isnull(article['full_text_file'])):

        summary['sections_summaries'] = "Full Text Is Not Acessible!"

        return summary

        

    full_text = get_full_text(article['title'], article['full_text_file'])

    sections = article_sections(full_text)

    

    for section in sections:

        number_of_sentences = len(sent_tokenize(section[1]))

        if(number_of_sentences > 1):

            # 1 Preprocessing Sections' Paraghraphs

            paraghraph_without_links = re.sub(r'^https?:\/\/.*[\r\n]*', '', section[1], flags=re.MULTILINE)

            # Removing doi urls from articles

            preprocessed_text  = re.sub(r'https://doi.org/10.1101/752592', '', paraghraph_without_links, flags=re.MULTILINE)

            section_summary =  gensim.summarization.summarize(preprocessed_text)

        else:

            section_summary = section[1]

            

        summary['sections_summaries'].append([section[0], section_summary])

    return summary
summarized_articles = []

for article in related_articles[:20]:

    summarized_articles.append(summarize_gensim(article))



for summary in summarized_articles:

    print("Title:")

    print(summary['title'])

    print("Publish Date:")

    print(summary['publish_time'])

    print("Summary:")

    if(summary['sections_summaries'] == "Full Text Is Not Acessible!"):

         print("Full Text Is Not Acessible!")

    else:

         for section in summary['sections_summaries']:

            print(section[0] + ":")

            print()

            print(section[1])

            print()

   

    print("-------------------------------------------------------------------------------------------")

    print('\n')

    