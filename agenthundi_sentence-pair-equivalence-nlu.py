# Getting all the required imports.

import pandas as pd

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, RegexpTokenizer

import string

import math

import numpy as np

import scipy as sp

from collections import namedtuple

import gensim

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import KeyedVectors

from gensim.models import Doc2Vec,Word2Vec
# Loading the dataset. Viewing its head

train = pd.read_csv('../input/qa_train_final.csv')

train.head()
train.is_duplicate.value_counts()
# Assigning the tokenizer

tokenizer = RegexpTokenizer("[\w']+")
# Applying the tokenizing function on the dataset

train['question1'] = train['question1'].apply(str)

train['question2'] = train['question2'].apply(str)

train['que_1_tok'] = train['question1'].apply(lambda x : tokenizer.tokenize(x))

train['que_2_tok'] = train['question2'].apply(lambda x : tokenizer.tokenize(x))
# Here we are generating three basic features from teh data at hand.

# 1. Grtting the length of sentences for question 1

# 2. Getting the length of sentences for question 2

# 3. Getting the length difference

train['q1_len'] = train['que_1_tok'].apply(lambda x: len(x))

train['q2_len'] = train['que_2_tok'].apply(lambda x: len(x))

train['len_diff'] = abs(train.q1_len - train.q2_len)
# Defining the function for calculating the common words, and normalizing them with the length of the questions,.

def normalized_common_words(df):

    w1 = set(map(lambda word: word.lower().strip(), df['question1'].split(" ")))

    w2 = set(map(lambda word: word.lower().strip(), df['question2'].split(" ")))    

    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))





train['word_share'] = train.apply(normalized_common_words, axis=1)
# Defining a function to remove the stopwords and removing any non-numeric words.

def rem_stpwrds(x):

    

    x = [word for word in x if word not in stopwords.words('english')]

    x = [word for word in x if word.isalpha() == True]

    

    return x
# Implementing the remove_stopwords function.

train['que_1_stp_wrds'] = train.que_1_tok.apply(lambda x: rem_stpwrds(x))

train['que_2_stp_wrds'] = train.que_2_tok.apply(lambda x: rem_stpwrds(x))
# Here we are generating three basic features from teh data at hand.

# 1. Grtting the length of sentences for question 1

# 2. Getting the length of sentences for question 2

# 3. Getting the length difference 

train['q1_stpwrds_len'] = train['que_1_stp_wrds'].apply(lambda x: len(x))

train['q2_stpwrds_len'] = train['que_2_stp_wrds'].apply(lambda x: len(x))

train['len_diff_stpwrds'] = abs(train.q1_stpwrds_len - train.q2_stpwrds_len)
# Dropping any Null values

train.dropna(inplace=True)
# A function to Check if the first word of the questions are same.

def prefix_match(row):

    

    for i in range(len(train)):

        

        b1 = str(row['que_1_tok'][i][0])

        b2 = str(row['que_2_tok'][i][0])    



        if b1 == b2:

            return 1

    

        else:

            return 0 



# Applying the function.

train['prefix_match'] = train.apply(prefix_match, axis=1)
# A function to check if the last words are similar.

def last_word_match(row):

    

    for i in range(len(train)):

        

        b1 = str(row['que_1_stp_wrds'][i][-1])

        b2 = str(row['que_2_stp_wrds'][i][-1])    



        if b1 == b2:

            return 1

    

        else:

            return 0 



# Applying the function.

train['last_word_match'] = train.apply(last_word_match, axis=1)
# Building a TF_IDF Model. Here are some of its attributes:

# min_df = Removes the vocabulary which has its count less than 2 times accross all the questions.

# analyzer = It takes a single word for creating the TF/IDF. 1-gram

# token_pattern = It considers the string.

# ngram_range = giving teh ngram range, so it uses uni and bi gram approach.

# smooth_idf = this prevents the dividing by zero error. 1 - laplace smoothing.

tf_idf_vecs = TfidfVectorizer(min_df=2,  max_features=None, strip_accents='unicode',  

      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), 

      use_idf=1,smooth_idf=1,sublinear_tf=1)

            

# Fit TFIDF    

tf_idf_vecs.fit(pd.concat([train['question1'],train['question2']]))



# fit the Tf-IDF model on teh questions and transform the questions into vectors.

tfidf_que1_vecs = tf_idf_vecs.transform(train['question1']) 

tfidf_que2_vecs = tf_idf_vecs.transform(train['question2'])
# Printing the shapes of output TF-IDF Vectors.

print (tfidf_que1_vecs.shape)

print (tfidf_que2_vecs.shape)
train.question1[5]
train.question2[5]
# Building a Word2Vec Model for Word Embeddings of the data. Some of its attributes are:

# size: It is the size of the output vector for each word.

# alpha: It is the learning rate

model_w2v = Word2Vec(list(train.que_1_stp_wrds) + list(train.que_2_stp_wrds),

    size=300,

    alpha=0.025,

    window=5,

    min_count=5) 



# Saving the model.

model_w2v.save('w2v.model')
# The generated Model

model_w2v
# This is the vector size of each word

model_w2v.vector_size
# Length of the unique vectors in teh vocabulary

len(model_w2v.wv.vocab)
# Getting the most similar word for each word in teh questions.

model_w2v.most_similar('What')
# Getting the questions and qids into a single dataset.

train_set1 = train[["qid1","question1"]]

train_set2 = train[["qid2","question2"]]



train_set1.columns = ["qid","question"]

train_set2.columns =["qid","question"]

train_set = pd.concat([train_set1,train_set2],axis=0)
# Initializing the required lists for storing teh que id with the question tagged to it. The input style for the Doc2vec

alldocuments = []

analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')       

keywords = []



# Method to built a tagged document with qid as tag id and que as the sentence. 

# This model dosent consider teh stopwords.

for index,record in train_set[:].iterrows():

    qid = str(record["qid"])

    question = str(record["question"])

    question = tokenizer.tokenize(question)

    #tokens = rem_stpwrds(question)

    #print(tokens)

    words_text = " ".join(question)

    words = gensim.utils.simple_preprocess(words_text)

    tags = [qid]

    alldocuments.append(analyzedDocument(words, tags))
# Parameters for Doc2vec model building. Max iterations is 100, learning rate is 0.025

# the vector size for each sentence being 300.

max_epochs = 100

vec_size = 300

alpha = 0.025



# Here we are decreasing the learning rate linearly.

model = Doc2Vec(size=vec_size,

                window=5, 

                seed=1337,

                workers=4,

                alpha=alpha, 

                min_alpha=0.00025,

                min_count=1,

                dm =0)



# Building the Doc2vec model.

model.build_vocab(alldocuments)



for epoch in range(max_epochs):

    #print('iteration {0}'.format(epoch))

    model.train(alldocuments,

                total_examples=model.corpus_count,

                epochs=model.iter)

    # decrease the learning rate

    model.alpha -= 0.0002

    # fix the learning rate, no decay

    model.min_alpha = model.alpha



# Saving the model

model.save("d2v_sentences.model")

print("Model Saved")
# This is to not build the model again and just trying to load the model.

d2v_model = Doc2Vec.load('d2v_sentences.model')
# Function to calculate the vector similarity of questions using Doc2Vec model. (Without Stopwords)

def cos_dist_d2v_stpwrds(df):

            

    vec1 = d2v_model.infer_vector(df['que_1_stp_wrds'])

    vec2 = d2v_model.infer_vector(df['que_2_stp_wrds'])

        

    cdist = sp.spatial.distance.cosine(vec1,vec2)

        

    return cdist
# Function to calculate the vector similarity of questions using Doc2Vec model. (With Stopwords)

def cos_dist_d2v(df):

            

    vec1 = d2v_model.infer_vector(df['que_1_tok'])

    vec2 = d2v_model.infer_vector(df['que_2_tok'])

            

    cdist = sp.spatial.distance.cosine(vec1,vec2)

        

    return cdist
# Applying the above methods to generate the cosine distances.

train['cosine_dist_d2v_stpwrds'] = train.apply(cos_dist_d2v_stpwrds, axis=1)

train['cosine_dist_d2v'] = train.apply(cos_dist_d2v, axis=1)

train.tail(50)
train.columns
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score, accuracy_score, recall_score, classification_report

from sklearn.model_selection import train_test_split
# Assigning the data and splitting into train and test.

X = tfidf_que1_vecs + tfidf_que2_vecs

y = train['is_duplicate']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)



# Building the log model and fitting.

model = LogisticRegression()

model.fit(X_train,y_train)

model_pred = model.predict(X_test)



# Calculating the model Evaluation metrics.

accuracy = accuracy_score(y_test,model_pred)

precision = precision_score(y_test,model_pred)

recall = recall_score(y_test,model_pred)

print('The accuracy is: ',accuracy)

print('The precision is: ',precision)

print('The recall is: ',recall)

print('\n')

print(classification_report(y_test,model_pred))
# Assigning the data and splitting into ttrain and test.

X = train[['q1_len', 'q2_len', 'len_diff', 'word_share','q1_stpwrds_len', 'q2_stpwrds_len',

       'len_diff_stpwrds', 'prefix_match', 'last_word_match',

       'cosine_dist_d2v_stpwrds', 'cosine_dist_d2v']]

y = train['is_duplicate']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)



# Building teh log model and fitting.

model = LogisticRegression()

model.fit(X_train,y_train)

model_pred = model.predict(X_test)



# Calculating the model Evaluation metrics.

accuracy = accuracy_score(y_test,model_pred)

precision = precision_score(y_test,model_pred)

recall = recall_score(y_test,model_pred)

print('The accuracy is: ',accuracy)

print('The precision is: ',precision)

print('The recall is: ',recall)

print('\n')

print(classification_report(y_test,model_pred))
from sklearn.naive_bayes import MultinomialNB

#spam_detect_model = MultinomialNB().fit(, messages['label'])



# Assigning the data and splitting into train and test.

X = tfidf_que1_vecs + tfidf_que2_vecs

y = train['is_duplicate']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)



# Building the log model and fitting.

model = MultinomialNB()

model.fit(X_train,y_train)

model_pred = model.predict(X_test)



# Calculating the model Evaluation metrics.

accuracy = accuracy_score(y_test,model_pred)

precision = precision_score(y_test,model_pred)

recall = recall_score(y_test,model_pred)

print('The accuracy is: ',accuracy)

print('The precision is: ',precision)

print('The recall is: ',recall)

print('\n')

print(classification_report(y_test,model_pred))