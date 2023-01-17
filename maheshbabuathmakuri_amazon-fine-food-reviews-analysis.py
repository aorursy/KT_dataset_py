# Importing the libraries

import numpy as np # Used for Numerical operations on data

import pandas as pd # for reading input files by using CSV/sqlite

import sqlite3 # to connect sqlite3 db

import nltk # Natural loanguage tool kit

from nltk.corpus import stopwords

import re # regular expression used to replace/Substitute specific words/pattern

from bs4 import BeautifulSoup # Used to replace html tags

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec





connection = sqlite3.connect("../input/amazon-fine-food-reviews/database.sqlite") # connecting to data

# read the data from sqlite3 db.

# Instead of reading 568,454 records, we are reading only 5000 records for early results.

df = pd.read_sql_query("SELECT * FROM Reviews where Score != 3 LIMIT 5000", connection) 

df.head(5)
df["Score"].value_counts()

# Data is imbalanced dataset as we have different number of categories
# Task is to find the review is positive or negative.

# We are considering if the review is less than 3 it is negative and more than 3 the review is positive 

def mappingfunction(x):

    if x < 3:

        return 0

    return 1



df["Score"] = df["Score"].map(mappingfunction)

df.head(5)
# Sorting the data based on Product Id

sorted_data = df.sort_values("ProductId", axis=0, na_position="last")

# Remove the dulicates if the UserId, ProfileName, Time, Text have the same values.

# Because one user can give only one review at particular time.

final_df = sorted_data.drop_duplicates(subset=["UserId","ProfileName","Time","Text"])

print("After removal duplicates we have {} number of reviews".format(final_df.shape[0]))
# removing the records if we have more number of Numerator than denominator

final_df = final_df.loc[final_df["HelpfulnessNumerator"] <= final_df["HelpfulnessDenominator"]]

print(final_df.shape[0])
# NLP

sent_0 = df["Text"].values[0]

print(sent_0)



# remove urls

sentence = re.sub(r"http\S+", "", sent_0)



# remove html tags

soup = BeautifulSoup(sentence, "lxml")

sentence = soup.get_text()





# Expand contraction words

def decontracted(phrase):

     # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase



sentence = decontracted(sentence)



# remove alpha numeric words(words with numbers)

sentence = re.sub("\S*\d\S*", "", sentence).strip()



# remove special characters

sentence = re.sub("[^A-Za-z0-9]+", " ", sentence)



# remove stopwords. Basically stopwords contains not aswell. but inorder to find the polarity of the review we need the not. 

# So we have created our wen set of stopwords.

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])



print(sentence)

# Applying all the above NLP techniques to reviews

from tqdm import tqdm # Creates progress bar and show how much percentage is completed

preprocessed_reviews = []

for sentence in tqdm(final_df["Text"].values):

    sentence = re.sub(r"http\S+", "", sentence)

    sentence = BeautifulSoup(sentence, 'lxml').get_text()

    sentence = decontracted(sentence)

    sentence = re.sub("\S*\d\S*", "", sentence).strip()

    sentence = re.sub('[^A-Za-z]+', ' ', sentence)

    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)

    preprocessed_reviews.append(sentence.strip())
# CountVectorizer: In this, We create set of unique words for all the text and 

# we fill the each cell of a review with the occurenrence of number of times of the word.

#BOW

count_vect = CountVectorizer()

count_vect.fit(preprocessed_reviews)

final_counts = count_vect.transform(preprocessed_reviews)

print(final_counts.shape)
# Basically Countvectorizer creates each dimension by taking each word into consideration and it is called uni-gram.

# Bi-gram: it takes 2 pair of words for creating a each dimension.

# if we create Countvectorizer by using ngram_range=(1,2) means: it creates the dimensions for both Unigram & bi-gram.

# ngram_range=(2,2) means: it creates dimension by using bi-grams only.

count_vect = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)

count_vect.fit(preprocessed_reviews)

final_bigram_counts = count_vect.transform(preprocessed_reviews)

print(final_bigram_counts.shape)

print("After Count Vectorizer, we have {0} rows and {1} columns".format(final_bigram_counts.shape[0], final_bigram_counts.shape[0]))

print("We have {0} unique words".format(final_bigram_counts.shape[0]))
# TfidfVectorizer

# ---------------

# TF(Term-Frequency): Number of times of occurences of word/ total number of word

# IDF(Inverse Document Frequency): logarthemic(Total Number of reviews/Number of reviews in which the word contains)



tf_idf_vect = TfidfVectorizer()

tf_idf_vect.fit(preprocessed_reviews)

final_tf_idf_vect = tf_idf_vect.transform(preprocessed_reviews)

print(final_tf_idf_vect.shape)
# Both Count Vectorizer and TfIdf Vectorizer creates the vector by considering occurences of the word 

# and it doesnot take semantic meaning into consideration.



# Word2Vec creates the d-dimenstional vector by considering semantic meaning.

# We used google genism model provided by Google for creating the 50-dimensional vector for each word.





# Implementation of word2 vector

# W2V takes list of lists into consideration for creating word2vector model. 

# So we split the reviews into list of lists(words) in below step

list_of_sentence = []

for sentence in preprocessed_reviews:

    list_of_sentence.append(sentence.split())



#print(list_of_sentence[0])



w2v_model = Word2Vec(list_of_sentence, min_count=5, size=50, workers=8) 

# min_count=5: To create the vector the word should appear atleast 5 times inthe document.

# size=50: Creates 50 diensional vector.





print(w2v_model.wv.most_similar("great"))



#print(list(w2v_model.wv.vocab))

#Avg-W2V



# Avg-W2V is sum of all the word 2 vectos for each word divided by number of words in the review

w2v_words = list(w2v_model.wv.vocab)

avg_w2v = []

for sentence in tqdm(list_of_sentence):

    sent_vec = np.zeros(50)

    cnt_words = 0

    for word in sentence:

        if word in w2v_words:

            word_vec = w2v_model.wv[word]

            sent_vec += word_vec

            cnt_words+=1

    if cnt_words != 0:

        word_vec = word_vec/len(sentence)

        avg_w2v.append(word_vec)

    



print(len(avg_w2v))

# TFIDF weighted W2v



# TFIDF weighted W2v: sum of product of tf-idf and w2v divided by sum of tf-idf

# formulae: (sum(tf-idf * w2v))/(sum(tf-idf))

model = TfidfVectorizer()

model.fit(preprocessed_reviews)



# Create dictionay with key: word and vlue: idf

tf_idf_dict = dict(zip(model.get_feature_names(), model.idf_))



# print("IDF's for the words: ", tf_idf_dict)
# Implementation

tf_idf_weighted_w2v = []

tfidf_feat = model.get_feature_names()

for sentence in tqdm(list_of_sentence):

    weight_sum = 0

    tf_idf_w2v = np.zeros(50)

    for word in sentence:

        if word in w2v_words and word in tfidf_feat:

            tf = sentence.count(word)/len(sentence)

            tf_idf = tf*tf_idf_dict[word]

            w2v = w2v_model.wv[word]

            tf_idf_w2v += (tf_idf*w2v)

            weight_sum += tf_idf

    if weight_sum != 0:

        tf_idf_w2v /= weight_sum

    tf_idf_weighted_w2v.append(tf_idf_w2v)

    