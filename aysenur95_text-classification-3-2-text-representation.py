import pickle

import pandas as pd

import numpy as np

import string

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import chi2

import re

import gzip
#load the pickle file for encoder le object

with open("../input/text-classification-2-feature-engineering/le.pkl", 'rb') as data:

    le = pickle.load(data)

    
#encoder dict

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

le_name_mapping
#load the pickle files for train&test sets

with open("../input/text-classification-2-feature-engineering/df_train.pkl", 'rb') as data:

    df_train = pickle.load(data)

    

with open("../input/text-classification-2-feature-engineering/df_test.pkl", 'rb') as data:

    df_test = pickle.load(data)
#labels

y_train=df_train['condition']

y_test=df_test['condition']
#tf-idf is similar to bow(formula: tf(t,d)*idf(t))



tfidf = TfidfVectorizer(encoding='utf-8',

                        ngram_range=(1,2),

                        lowercase=False,

                        max_df=1.0, #100&

                        min_df=10,  #10

                        max_features=1000,

                        sublinear_tf=True)

                        

X_train_tfidf = tfidf.fit_transform(df_train['review_parsed']).toarray()



X_test_tfidf = tfidf.transform(df_test['review_parsed']).toarray()

#see what unigrams and bigrams are most correlated with each category.

for condition, condition_id in le_name_mapping.items():

    features_chi2 = chi2(X_train_tfidf, y_train==condition_id)

    indices = np.argsort(features_chi2[0])

    sorted_feature_names = np.array(tfidf.get_feature_names())[indices]

    unigrams = [feat for feat in sorted_feature_names if len(feat.split(' ')) == 1]

    bigrams = [feat for feat in sorted_feature_names if len(feat.split(' ')) == 2]

    print("###### '{}' category ######".format(condition))

    print("  --> Most correlated unigrams:\n+ {}".format('\n+ '.join(unigrams[-5:])))

    print("  --> Most correlated bigrams:\n+ {}".format('\n+ '.join(bigrams[-5:])))

    print("")

    print("************************************")

    print("")
# y_train 

with gzip.open('y_train.pkl', 'wb') as output:

    pickle.dump(y_train, output, protocol=-1)

     

    

# y_test

with gzip.open('y_test.pkl', 'wb') as output:

    pickle.dump(y_test, output, protocol=-1)   

    

    

# x_train_tfidf    

with gzip.open('x_train_tfidf.pkl', 'wb') as output:

    pickle.dump(X_train_tfidf, output, protocol=-1)

    

    

# x_test_tfidf    

with gzip.open('x_test_tfidf.pkl', 'wb') as output:

    pickle.dump(X_test_tfidf, output, protocol=-1)  

    

        

# tfidf

with gzip.open('tfidf.pkl', 'wb') as output:

    pickle.dump(tfidf, output, protocol=-1)

    