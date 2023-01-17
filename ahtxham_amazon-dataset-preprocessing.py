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

Amazon_Meta_Data = pd.read_csv('../input/amazon-reviews-unlocked-mobile-phones/Amazon_Unlocked_Mobile.csv',nrows=1000, encoding='utf-8')



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
Reviews = Amazon_Meta_Data['Reviews']

Brand_Name = Amazon_Meta_Data['Brand Name'].str.upper()

brand_list =list(set(Brand_Name))

unique_products=list(set(Amazon_Meta_Data['Product Name']))

unique_rating = list(set(Amazon_Meta_Data['Rating']))



#**unique products in dataset and total reviews about each product in dataset.**

product_count = []

for prod in unique_products:

    product_count.append(len(Amazon_Meta_Data.loc[Amazon_Meta_Data['Product Name'] == prod]))

    



#**Detailed information About dataset**

# print('total Reviews    : ',len(Reviews))

# print('total Brands     : ',len(Brand_Name.value_counts()))

# print('total Products   : ',len(product_count))

# print('total Brands with their counts are \n',Brand_Name.value_counts())





#**Ratings count according to the product**

rating=list(set(Amazon_Meta_Data['Rating']))

for rating in unique_rating:

    globals()['rating_list_%s' % rating] =[]

for prod in unique_products:

    for rating in unique_rating:

        globals()['rating_list_%s' % rating].append(len(Amazon_Meta_Data.loc[(Amazon_Meta_Data['Product Name'] == prod) & (Amazon_Meta_Data['Rating']==rating)]))



        

products_dataset = pd.DataFrame()

products_dataset['Product Name'] = unique_products

products_dataset['Total_Reviews'] = product_count

products_dataset['1_Rating'] = rating_list_1

products_dataset['2_Rating'] = rating_list_2

products_dataset['3_Rating'] = rating_list_3

products_dataset['4_Rating'] = rating_list_4

products_dataset['5_Rating'] = rating_list_5



# products_dataset.head()
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

    

    

#----------------------  extract Sentences  -------------------------    

def sentances(text):

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    return sent_detector.tokenize(text.strip())

#-----------------  Text Cleaning & Sentiment Extraction --------------------

cleaned_reviews =[]

sentiment= []

for reviews in Amazon_Meta_Data['Reviews']:

    cleaned_reviews.append(clean_text(reviews))

    sentiment.append(get_text_sentiment(reviews))

    

features_Dataset = pd.DataFrame()

features_Dataset['Product Name'] = Amazon_Meta_Data['Product Name']

features_Dataset['Reviews'] = Amazon_Meta_Data['Reviews']

features_Dataset['Cleaned_Reviews'] = cleaned_reviews

features_Dataset['Sentiment'] = sentiment





# Extracting Features from each review using TextBlob & Spacy *

    

#--------------------- Text Blob -------------------------

features = []

for reviews in features_Dataset['Cleaned_Reviews']:

    features.append(textBlob_feature_extraction(reviews))

#adding Extracted features to dataset

features_Dataset["TextBlob_Features"] = features



#--------------------- Spacy -----------------------------

nlp = spacy.load('en')



feature_spacy = []

for review in nlp.pipe(features_Dataset['Cleaned_Reviews']):

    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']

    feature_spacy.append(','.join(chunks))



features_Dataset['Spacy_features']= feature_spacy



#---------------------------------------------------------

features_Dataset.head()



#----------------- Features Counts ------------------

#function to extract Features With their counts

counts = {}

def feature_count(feature):

    if  feature in counts:

        counts[feature] += 1

    else:

        counts[feature] = 1

#Extracting feature count form feartures

for feature_list in features_Dataset['TextBlob_Features']:

    for feature in feature_list:

        feature_count(feature)



#-----------------Distinct Features Counts ------------------

#dataframe of features and their count    

dist_feature =[]

dist_feature_count= []

for key, value in counts.items() :

    dist_feature.append(key)

    dist_feature_count.append(value)

    

#----------------- Features Counts Dataset ------------------

feature_TextBlob = pd.DataFrame()

feature_TextBlob['Feature Name'] =dist_feature

feature_TextBlob['Feature Count'] =dist_feature_count



#----------------- Features Details ------------------

#discarding feature that occure less the 3 in the entire dataset

print('TextBlob features extracted from dataset are : ' ,len(feature_TextBlob))

feature_TextBlob =feature_TextBlob.loc[feature_TextBlob['Feature Count'] >=3]

print('TextBlob features left after discarding those who occure less then three times in dataset are : ' ,len(feature_TextBlob))

#feature_TextBlob



#--------------------  Downlaodable CVS of Features and their count --------------------

from IPython.display import HTML

feature_TextBlob.to_csv('feature_TextBlob.csv', index=False)



def create_download_link(title = "Download features and counts as CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='feature_TextBlob.csv')

##===========================================================================================================

module_two = pd.DataFrame()

module_two["Products"] = Amazon_Meta_Data["Product Name"]

module_two["Reviews"] = Amazon_Meta_Data["Reviews"]

module_two = module_two.loc[(module_two['Products'] == '4 Inch Touch Screen Cell Phone Unlocked, Android Unlocked Gsm Smartphones Dual Camera Dual Sim Dual Standby No Contract (Blue)')]
sentiment_sent= []

sentences = []

for reviews in module_two["Reviews"]:

    for sent in sentances(reviews):

        sentiment_sent.append(get_text_sentiment(sent))

        sentences.append(sent)





prod1_dataset = pd.DataFrame()

prod1_dataset["Reviews_sentences"] =  sentences

prod1_dataset["sentiment"] = sentiment_sent



prod1_dataset.head(10)

cleaned_reviews =[]

sentiment= []

for reviews in prod1_dataset["Reviews_sentences"]:

    cleaned_reviews.append(clean_text(reviews))

    sentiment.append(get_text_sentiment(reviews))





New_df = pd.DataFrame()

New_df['Reviews_sentences'] = prod1_dataset["Reviews_sentences"]

New_df['Cleaned_Reviews_sentences'] = cleaned_reviews

New_df['Sentiment'] = sentiment





New_df.head()



# Extracting Features from each review using TextBlob **

features = []

for reviews in New_df['Cleaned_Reviews_sentences']:

    features.append(textBlob_feature_extraction(reviews))

#adding Extracted features to dataset

New_df["TextBlob_Features"] = features

New_df.head(10)

#dataset.dtypes

positive_reviews = New_df.loc[(New_df['Sentiment'] == 'positive')]

positive_reviews.head(10)
#----------------- Features Counts ------------------

#function to extract Features With their counts

counts = {}

def feature_count(feature):

    if  feature in counts:

        counts[feature] += 1

    else:

        counts[feature] = 1

#Extracting feature count form feartures

for feature_list in positive_reviews['TextBlob_Features']:

    for feature in feature_list:

        feature_count(feature)



#-----------------Distinct Features Counts ------------------

#dataframe of features and their count    

dist_feature =[]

dist_feature_count= []

for key, value in counts.items() :

    dist_feature.append(key)

    dist_feature_count.append(value)

    

#----------------- Features Counts Dataset ------------------

positive_reviews_features = pd.DataFrame()

positive_reviews_features['Feature Name'] =dist_feature

positive_reviews_features['Feature Count'] =dist_feature_count



#----------------- Features Details ------------------

#discarding feature that occure less the 3 in the entire dataset

print('TextBlob features extracted from dataset are : ' ,len(positive_reviews_features))

##positive_reviews_features =positive_reviews_features.loc[positive_reviews_features['Feature Count'] >=3]

#print('TextBlob features left after discarding those who occure less then three times in dataset are : ' ,len(positive_reviews_features))

#feature_TextBlob



positive_reviews_features.head(10)

##===========================================================================================================
negative_reviews = New_df.loc[(New_df['Sentiment'] == 'negative')]

negative_reviews.head(10)
#----------------- Features Counts ------------------

#function to extract Features With their counts

counts = {}

def feature_count(feature):

    if  feature in counts:

        counts[feature] += 1

    else:

        counts[feature] = 1

#Extracting feature count form feartures

for feature_list in negative_reviews['TextBlob_Features']:

    for feature in feature_list:

        feature_count(feature)



#-----------------Distinct Features Counts ------------------

#dataframe of features and their count    

dist_feature =[]

dist_feature_count= []

for key, value in counts.items() :

    dist_feature.append(key)

    dist_feature_count.append(value)

    

#----------------- Features Counts Dataset ------------------

negative_reviews_features = pd.DataFrame()

negative_reviews_features['Feature Name'] =dist_feature

negative_reviews_features['Feature Count'] =dist_feature_count



#----------------- Features Details ------------------

#discarding feature that occure less the 3 in the entire dataset

#print('TextBlob features extracted from dataset are : ' ,len(negative_reviews_features))

#print('TextBlob features left after discarding those who occure less then three times in dataset are : ' ,len(negative_reviews_features))

#feature_TextBlob



negative_reviews_features.head(10)

##===========================================================================================================

product_count = []

for prod in unique_products:

    product_count.append(len(Amazon_Meta_Data.loc[Amazon_Meta_Data['Product Name'] == prod]))

Aspects_polarity_DB = pd.DataFrame()

Aspects_polarity_DB['Product Name'] =unique_products

Aspects_polarity_DB['Total Reviews'] =product_count

print('total products Before descarding less then 5 reviews products',len(Aspects_polarity_DB))

Aspects_polarity_DB=Aspects_polarity_DB.loc[Aspects_polarity_DB['Total Reviews'] >=5]

print('total products after descarding less then 5 reviews products',len(Aspects_polarity_DB))