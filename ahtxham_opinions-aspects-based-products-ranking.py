import pandas as pd

import numpy as np

from textblob import TextBlob



import nltk

from nltk import pos_tag

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import LancasterStemmer, WordNetLemmatizer

import re, string, unicodedata

#import inflect



import spacy

#nltk.download('stopwords')



import nltk.data



import os

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

#from __future__ import unicode_literals



import warnings

warnings.filterwarnings("ignore")

Amazon_Meta_Data = pd.read_csv('../input/amazon-reviews-unlocked-mobile-phones/Amazon_Unlocked_Mobile.csv',nrows=50000, encoding='utf-8')



#------- Counting words in reviews ---------

word_counts = []

for review in Amazon_Meta_Data['Reviews']:

    count=0

    for word in str(review).split():

        count +=1

    word_counts.append(count)

Amazon_Meta_Data['words_counts'] = word_counts



# Discarding rows having less then 5 words of review.

Amazon_Meta_Data=Amazon_Meta_Data.loc[Amazon_Meta_Data['words_counts'] >=5]



Amazon_Meta_Data.head()

#Amazon_Meta_Data.dtypes
def remove_non_ascii(words):

    """Remove non-ASCII characters from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        new_words.append(new_word)

    return new_words



def to_lowercase(words):

    """Convert all characters to lowercase from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = word.lower()

        new_words.append(new_word)

    return new_words



def remove_punctuation(words):

    """Remove punctuation from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = re.sub(r'[^\w\s]', '', word)

        if new_word != '':

            new_words.append(new_word)

    return new_words



def replace_numbers(words):

    """Replace all interger occurrences in list of tokenized words with textual representation"""

    p = inflect.engine()

    new_words = []

    for word in words:

        if word.isdigit():

            new_word = p.number_to_words(word)

            new_words.append(new_word)

        else:

            new_words.append(word)

    return new_words



def remove_stopwords(words):

    """Remove stop words from list of tokenized words"""

    new_words = []

    for word in words:

        if word not in stopwords.words('english'):

            new_words.append(word)

    return new_words



def stem_words(words):

    """Stem words in list of tokenized words"""

    stemmer = LancasterStemmer()

    stems = []

    for word in words:

        stem = stemmer.stem(word)

        stems.append(stem)

    return stems



def lemmatize_verbs(words):

    """Lemmatize verbs in list of tokenized words"""

    lemmatizer = WordNetLemmatizer()

    lemmas = []

    for word in words:

        lemma = lemmatizer.lemmatize(word, pos='v')

        lemmas.append(lemma)

    return lemmas



def normalize(words):

    words = remove_non_ascii(words)

    words = to_lowercase(words)

    words = remove_punctuation(words)

    #words = replace_numbers(words)

    words = remove_stopwords(words)

    return words



#Steemming and Lemmatization

def stem_and_lemmatize(words):

    stems = stem_words(words)

    lemmas = lemmatize_verbs(words)

    return stems, lemmas



#stems, lemmas = stem_and_lemmatize(words)

#print('Stemmed:\n', stems)

#print('\nLemmatized:\n', lemmas)





# ---------------   Cleaning   ------------------

def clean_text(text):

    wording = nltk.word_tokenize(text)

    words = normalize(wording)

    string_text = ' '.join(words)

    return string_text



#----------------------  extract Sentences  -------------------------    

def sentances(text):

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    return sent_detector.tokenize(text.strip())





# ---------------   Sentiment   ------------------

def get_text_sentiment(text):

    # create TextBlob object of passed text 

    analysis = TextBlob(clean_text(text)) 

    # set sentiment 

    if analysis.sentiment.polarity > 0: 

        return 'positive'

    elif analysis.sentiment.polarity == 0: 

        return 'neutral'

    else: 

        return 'negative'    

    

#---------------------- TextBlob Feature Extractions -----------------

#Function to extract features from text

def textBlob_feature_extraction(text): 

        blob = TextBlob(text)

        return blob.noun_phrases
sentences = []

sentiment_sent =[]

product_name =[]

for products,reviews in zip(Amazon_Meta_Data["Product Name"], Amazon_Meta_Data["Reviews"]): 

    for sent in sentances(reviews):

        sentiment_sent.append(get_text_sentiment(sent))

        sentences.append(sent)

        product_name.append(products)



all_products_dataset = pd.DataFrame()

all_products_dataset["products"] = product_name

all_products_dataset["reviews_sentences"] =  sentences

all_products_dataset["sentiment"] = sentiment_sent



#Removing Neural sentences

all_products_dataset = all_products_dataset.loc[(all_products_dataset['sentiment'] != 'neutral')]



# Text Cleaning of sentences

cleaned_reviews =[]

for reviews in all_products_dataset["reviews_sentences"]:

    cleaned_reviews.append(clean_text(reviews))

    

#inserting Cleaned reviews sentences to our dataset

all_products_dataset.insert(2, 'cleaned_reviews_sentences ', cleaned_reviews)



#Extracting Features From Reviews Sentences

features = []

for reviews in cleaned_reviews:

    features.append(textBlob_feature_extraction(reviews))



all_products_dataset['texblob_features_extraction']=features
prod_features_counts = {}

for prod,feature_list, in zip(all_products_dataset['products'],all_products_dataset['texblob_features_extraction']):

    for feature in feature_list:

        if prod in prod_features_counts:

            if feature in prod_features_counts[prod]:

                prod_features_counts[prod][feature] += 1

            else:

                prod_features_counts[prod][feature] = 1 

        else:

            prod_features_counts[prod]={}



#Extracting feature count form feartures

dist_products =[]

dist_feature_count= []

for key, value in prod_features_counts.items() :

    dist_products.append(key)

    dist_feature_count.append(value)

    

#----------------- Features Counts Dataset ------------------

products_features_details = pd.DataFrame()

products_features_details['products'] = dist_products

products_features_details['feature with count'] = dist_feature_count

products_features_details.head(10)

positive_products_dataset = all_products_dataset.loc[(all_products_dataset['sentiment'] == 'positive')]



positive_features_counts = {}

for prod,feature_list, in zip(positive_products_dataset['products'],positive_products_dataset['texblob_features_extraction']):

    for feature in feature_list:

        if prod in positive_features_counts:

            if feature in positive_features_counts[prod]:

                positive_features_counts[prod][feature] += 1

            else:

                positive_features_counts[prod][feature] = 1 

        else:

            positive_features_counts[prod]={}



#Extracting feature count form feartures

dist_products =[]

dist_feature_count= []

for key, value in positive_features_counts.items() :

    dist_products.append(key)

    dist_feature_count.append(value)

    

#----------------- Features Counts Dataset ------------------

positive_features_details = pd.DataFrame()

positive_features_details['products'] = dist_products

positive_features_details['feature with count'] = dist_feature_count

positive_features_details.head(10)

negative_products_dataset = all_products_dataset.loc[(all_products_dataset['sentiment'] == 'negative')]



negative_features_counts = {}

for prod,feature_list, in zip(negative_products_dataset['products'],negative_products_dataset['texblob_features_extraction']):

    for feature in feature_list:

        if prod in negative_features_counts:

            if feature in negative_features_counts[prod]:

                negative_features_counts[prod][feature] += 1

            else:

                negative_features_counts[prod][feature] = 1 

        else:

            negative_features_counts[prod]={}

            

#Extracting feature count form feartures

dist_products =[]

dist_feature_count= []

for key, value in negative_features_counts.items() :

    dist_products.append(key)

    dist_feature_count.append(value)

    

#----------------- Features Counts Dataset ------------------

negative_features_details = pd.DataFrame()

negative_features_details['products'] = dist_products

negative_features_details['feature with count'] = dist_feature_count

negative_features_details.head(10)
#Removing Rows having null features

#all_products_dataset = all_products_dataset.loc[(all_products_dataset.astype(str)['texblob_features_extraction'] != '[]')]

#all_products_dataset
#user_specification = input("Search Here")

user_specification = 'good phone with good battery life and affordable price '

user_specification_sentiment = get_text_sentiment(user_specification)

user_specification_features = textBlob_feature_extraction(user_specification)

for feature in user_specification_features:

    print(feature)

    

    

#Assiging weights to products with max matching 

prod_weight = {}

if(user_specification_sentiment == 'positive'):

    for prod in positive_features_counts:

        for feature in user_specification_features:

            if prod in prod_weight:

                if feature in positive_features_counts[prod]:

                    prod_weight[prod] = prod_weight[prod]+ positive_features_counts[prod][feature]                    

            else:

                if feature in positive_features_counts[prod]:

                    prod_weight[prod] = positive_features_counts[prod][feature]





dist_feature =[]

dist_feature_count= []                    

for key, value in prod_weight.items() :

    dist_feature.append(key)

    dist_feature_count.append(value)

    

rank_products = pd.DataFrame()

rank_products['product'] =dist_feature

rank_products['weight'] =dist_feature_count

rank_products= rank_products.sort_values(by ='weight' , ascending=False)

rank_products             
#user_specification = input("Search Here")

user_specification = 'good battery life and affordable price and fast processing'

user_specification_sentiment = get_text_sentiment(user_specification)

user_specification_features = textBlob_feature_extraction(user_specification)

for feature in user_specification_features:

    print(feature)

    

    

#Assiging weights to products with max matching 

prod_weight = {}

if(user_specification_sentiment == 'positive'):

    for prod in positive_features_counts:

        for feature in user_specification_features:

            if prod in prod_weight:

                if feature in positive_features_counts[prod]:

                    prod_weight[prod] = prod_weight[prod]+ positive_features_counts[prod][feature]                    

            else:

                if feature in positive_features_counts[prod]:

                    prod_weight[prod] = positive_features_counts[prod][feature]





dist_feature =[]

dist_feature_count= []                    

for key, value in prod_weight.items() :

    dist_feature.append(key)

    dist_feature_count.append(value)

    

rank_products = pd.DataFrame()

rank_products['product'] =dist_feature

rank_products['weight'] =dist_feature_count

rank_products= rank_products.sort_values(by ='weight' , ascending=False)

rank_products             
#user_specification = input("Search Here")

user_specification = 'I need a phone with good price and good camera result '

user_specification_sentiment = get_text_sentiment(user_specification)

user_specification_features = textBlob_feature_extraction(user_specification)

for feature in user_specification_features:

    print(feature)

    

    

#Assiging weights to products with max matching 

prod_weight = {}

if(user_specification_sentiment == 'positive'):

    for prod in positive_features_counts:

        for feature in user_specification_features:

            if prod in prod_weight:

                if feature in positive_features_counts[prod]:

                    prod_weight[prod] = prod_weight[prod]+ positive_features_counts[prod][feature]                    

            else:

                if feature in positive_features_counts[prod]:

                    prod_weight[prod] = positive_features_counts[prod][feature]





dist_feature =[]

dist_feature_count= []                    

for key, value in prod_weight.items() :

    dist_feature.append(key)

    dist_feature_count.append(value)

    

rank_products = pd.DataFrame()

rank_products['product'] =dist_feature

rank_products['weight'] =dist_feature_count

rank_products= rank_products.sort_values(by ='weight' , ascending=False)

rank_products             
user_specification = input("Search Here")

#user_specification = 'I need a phone with good price and good camera result '

user_specification_sentiment = get_text_sentiment(user_specification)

user_specification_features = textBlob_feature_extraction(user_specification)

for feature in user_specification_features:

    print(feature)

    

    

#Assiging weights to products with max matching 

prod_weight = {}

if(user_specification_sentiment == 'positive'):

    for prod in positive_features_counts:

        for feature in user_specification_features:

            if prod in prod_weight:

                if feature in positive_features_counts[prod]:

                    prod_weight[prod] = prod_weight[prod]+ positive_features_counts[prod][feature]                    

            else:

                if feature in positive_features_counts[prod]:

                    prod_weight[prod] = positive_features_counts[prod][feature]





dist_feature =[]

dist_feature_count= []                    

for key, value in prod_weight.items() :

    dist_feature.append(key)

    dist_feature_count.append(value)

    

rank_products = pd.DataFrame()

rank_products['product'] =dist_feature

rank_products['weight'] =dist_feature_count

rank_products= rank_products.sort_values(by ='weight' , ascending=False)

rank_products             