# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import json
business = pd.read_csv("../input/yelp_business.csv")
review = pd.read_csv("../input/yelp_review.csv")
user = pd.read_csv("../input/yelp_user.csv")
business_NV = business[business['state'] == 'NV']
unique_ids = business_NV['business_id'].unique()
unique_id = unique_ids[1:200]
def review_business(business_id,type_of_review = 'positive'):
    review_for_business = review[review['business_id'] == business_id]
    if type_of_review == 'positive':
        type_reviews = review_for_business[review_for_business['stars']>3]
    elif type_of_review == 'negative':
        type_reviews = review_for_business[review_for_business['stars']<=3]
    return type_reviews

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
topic_modelling = {}

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

Lda = gensim.models.ldamodel.LdaModel
i = 0

for ids in unique_id:
    
    positive_topics = []
    negative_topics = []

    positive_reviews = review_business(ids)
    negative_reviews = review_business(ids)
    all_positive_reviews = []
    all_negative_reviews = []

    for no_of_reviews in positive_reviews['text']:
        all_positive_reviews.append(no_of_reviews)
        
    for no_of_reviews in negative_reviews['text']:
        all_negative_reviews.append(no_of_reviews)
        
    doc_clean_positive = [clean(doc).split() for doc in all_positive_reviews]
    doc_clean_negative = [clean(doc).split() for doc in all_negative_reviews]

    dictionary_positive = corpora.Dictionary(doc_clean_positive)
    dictionary_negative = corpora.Dictionary(doc_clean_negative)
    
    doc_term_matrix_positive = [dictionary_positive.doc2bow(doc) for doc in doc_clean_positive]
    doc_term_matrix_negative = [dictionary_negative.doc2bow(doc) for doc in doc_clean_negative]
    
    if doc_term_matrix_positive != []:
        ldamodel_positive = Lda(doc_term_matrix_positive, num_topics=3, id2word = dictionary_positive, passes=50)
        positive_topics.append(ldamodel_positive.print_topics(num_topics=3, num_words=3))
    else:
        positive_topics = []
        
    if doc_term_matrix_negative != []:
        ldamodel_negative = Lda(doc_term_matrix_negative, num_topics=3, id2word = dictionary_negative, passes=50)
        negative_topics.append(ldamodel_negative.print_topics(num_topics=3, num_words=3))
    else:
        negative_topics = []


    topics = {'positive_topics': positive_topics, 'negative_topics': negative_topics} 
    topic_modelling[ids] = topics

with open('result.json', 'w') as fp:
    json.dump(topic_modelling, fp)