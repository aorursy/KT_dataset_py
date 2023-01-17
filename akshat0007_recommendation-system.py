# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/flipkart_com-ecommerce_sample.csv")
df.head()
import re

import nltk

from nltk import pos_tag, word_tokenize, PorterStemmer

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

wordnet_lemmatizr=WordNetLemmatizer()

from termcolor import colored

def clean_product_type(dataframe):

    document=list(dataframe['product_category_tree'])

    product_types=[re.findall(r'\"(.*?)\"', sentence) for sentence in document]

    product_types=[' '.join(listed_items) for listed_items in product_types]

    return(product_types)
def clean_categories(dataframe):

    document=list(dataframe['product_category_tree'].values)

    categories=[re.findall(r'name=(.*?)}',sentence) for sentence in document]

    categories=[' '.join(word) for word in categories]

    return(categories)

def special_characters_cleaning(document):

    sentences=[]

    for sentence in document:

        sentences.append(re.sub('[^a-zA-Z0-9\n\.]',' ',str(sentence)))

    return(sentences)
def lemmetize_document(document):

    sentences=[]

    for sentence in document:

        word=[wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(sentence)]

        sentences.append(' '.join(words))

    return(sentences)
def categories_extraction(dataframe):

    categories=[word for item in dataframe['categories'] for word in item.split()]

    categories=list(set(categories))

    return(categories)
def save_categories(dataframe):

    pass

def pre_processing_document(document):

    document=special_characters_cleaning(document)

    document=lemmetize_document(document)

    document=[sentence.title() for sentence in document]

    return(document)
def extract_categories_from_description(document,categories):

    extracted_categories=[]

    for sentence in document:

        extracted_categories.append(' '.join(set(categories).intersection(set(word_tokenize(sentence)))))

        return(extracted_categories)
lemmetize= WordNetLemmatizer()

stemmer=PorterStemmer()
df["products"]=clean_product_type(df)
df["categories"]=clean_categories(df)
categories= list(set(df['product_category_tree'].values))

categories= [item.split() for item in df['product_category_tree']]

categories= [word.lower() for listed_item in categories for word in listed_item]

categories= list(set(categories))
df
df['detailed_description']= df['products']+ df['brand']+df['product_name']
df
document= list(df['detailed_description'].values)

document= special_characters_cleaning(document)
tfidf= TfidfVectorizer(stop_words= 'english', vocabulary= categories)

data= tfidf.fit_transform(document)
from sklearn.neighbors import NearestNeighbors

nn= NearestNeighbors(algorithm= 'brute', n_neighbors= 20).fit(data)
text= df[df['brand']== "FabHomeDecor"]['detailed_description'].values

result = nn.kneighbors(tfidf.transform(text))

for col in tfidf.transform(text).nonzero()[1]:

    print(tfidf.get_feature_names()[col], ' - ', tfidf.transform(text)[0, col])
for item in result[1][0]:

    print(colored(df.iloc[item]['product_category_tree'].upper(), 'blue'), ':', document[item])