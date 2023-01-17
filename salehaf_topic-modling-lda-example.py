!ls -a
'''

Load the dataset from the CSV and save it to 'data_text'

'''

import pandas as pd

data = pd.read_csv(r'../input/abcnews-date-text.csv', error_bad_lines=False);

# We only need the Headlines text column from the data

data_text = data[['headline_text']];
data.head(2)
'''

Add an index column to the dataset and save the dataset as 'documents'

'''

data_text['index'] = data_text.index

documents = data_text

documents
'''

Get the total number of documents

'''

print(len(documents))
'''

Preview a document and assign the index value to 'document_num'

'''

document_num = 4310

print("\n**Printing out a sample document:**")

print(documents[documents['index'] == document_num])
'''

Seperate the value of the headline from the document selected with 'document_num'

'''

print(documents[documents['index'] == document_num].values[0][0])
'''

Loading Gensim and nltk libraries

'''

# pip install gensim

import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import numpy as np

np.random.seed(400)
import nltk

#nltk.download()
'''

Lemmatizing example for a verb, noun.

'''

print(WordNetLemmatizer().lemmatize('went', pos = 'v')) # past tense to present tense
'''

Stemming example

'''

stemmer = SnowballStemmer("english")

plurals = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 

           'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 

           'traditional', 'reference', 'colonizer','plotted']

singles = [stemmer.stem(plural) for plural in plurals]

print(' '.join(singles))
'''

Write a function to perform the pre processing steps on the entire dataset

'''

def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))



# Tokenize and lemmatize

def preprocess(text):

    result = [lemmatize_stemming(token) for token in gensim.utils.simple_preprocess(text) 

              # Remove stop words and words less than 3 characters long

              if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3]

    return result
'''

Preview a document after preprocessing

'''



doc_sample = documents[documents['index'] == document_num].values[0][0]



print("Original document: ")

words = []

for word in doc_sample.split(' '):

    words.append(word)

print(words)

print("\n\nTokenized and lemmatized document: ")

print(preprocess(doc_sample))
'''

Save the values of the headlines into a variable 'training_headlines'

'''

training_headlines = [value[0] for value in documents.iloc[0:].values];
'''

Preview 'training_headlines'

'''

training_headlines
'''

Perform preprocessing on entire dataset using the function defined earlier and save it to 'processed_docs'

'''

processed_docs = [preprocess(doc) for doc in training_headlines]
'''

Preview 'processed_docs'

'''

processed_docs
'''

Create a dictionary containing the number of times a word appears in the training set and call it 'dictionary'

'''

dictionary = gensim.corpora.Dictionary(processed_docs)
'''

Checking dictionary created

'''

count = 0

for k, v in dictionary.iteritems():

    print(k, v)

    count += 1

    if count > 10:

        break
'''

Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many

words and how many times those words appear. Save this to 'bow_corpus'

'''

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
'''

Checking Bag of Words corpus for our sample document --> (token_id, token_count)

'''

# document_num = 4310

bow_corpus[document_num]
'''

Preview BOW for our sample preprocessed document

'''

# Here document_num is document number 4310 which we have checked in Step 2

bow_doc_4310 = bow_corpus[document_num]



for i in range(len(bow_doc_4310)):

    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 

                                                     dictionary[bow_doc_4310[i][0]], 

                                                     bow_doc_4310[i][1]))
'''

Create tf-idf model object on 'bow_corpus' and save it to 'tfidf'

'''

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
'''

Apply transformation to the entire corpus and call it 'corpus_tfidf'

'''

corpus_tfidf = tfidf[bow_corpus]
'''

Preview TF-IDF scores for our first document --> --> (token_id, tfidf score)

'''

from pprint import pprint

for doc in corpus_tfidf:

    pprint(doc)

    break
# LDA multicore 

'''

Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'

'''

lda_model = gensim.models.LdaMulticore(bow_corpus, 

                                       num_topics=10, 

                                       id2word = dictionary, 

                                       passes = 2, 

                                       workers=2)
'''

For each topic, we will explore the words occuring in that topic and its relative weight

'''

for idx, topic in lda_model.print_topics(-1):

    print("Topic: {} Word: {}".format(idx, topic))

    print("\n")
'''

Define lda model using tfidf corpus

'''

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, 

                                             num_topics=10, 

                                             id2word = dictionary, 

                                             passes = 2, 

                                             workers=4)
'''

For each topic, we will explore the words occuring in that topic and its relative weight

'''

for idx, topic in lda_model_tfidf.print_topics(-1):

    print("Topic: {} Word: {}".format(idx, topic))

    print("\n")
'''

Text of sample document 4310

'''

print(train_headlines[4310])
'''

Check which topic our test document belongs to using the LDA Bag of Words model.

'''

document_num = 4310

# Our test document is document number 4310

for index, score in sorted(lda_model[bow_corpus[document_num]], key=lambda tup: -1*tup[1]):

    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
'''

Check which topic our test document belongs to using the LDA TF-IDF model.

'''

document_num = 4310

# Our test document is document number 4310

for index, score in sorted(lda_model_tfidf[bow_corpus[document_num]], key=lambda tup: -1*tup[1]):

    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
unseen_document = "My favorite sports activities are running and swimming."



# Data preprocessing step for the unseen document

bow_vector = dictionary.doc2bow(preprocess(unseen_document))



for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):

    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))