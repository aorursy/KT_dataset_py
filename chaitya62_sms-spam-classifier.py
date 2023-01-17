 



import numpy as np 

import pandas as pd 

from collections import Counter

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

import nltk





df = pd.read_csv('../input/spam.csv',encoding='ISO-8859-1')

df['v1'] = df['v1'].map(lambda x: 1 if x =='ham' else 0)

df['words'] = df['v2'].map(lambda x: word_tokenize(x))

all_words = []

for i in df.words:

    all_words.extend(i)

stop_words = set(stopwords.words('english'))



all_words = [word for word in all_words if word not in stop_words]

all_words = nltk.FreqDist(all_words)



all_words_len = len(all_words)



# Considering the top 30 percent words as features

word_features = list(all_words)[:int(all_words_len*0.30)] 



records = list(zip(df.v1, df.words))



def get_features(sms):

    words = set(sms)

    features = {}

    for i in word_features:

        features[i] = (i in words)

    return features



# Prepare data for training

features_set = [(get_features(sms),category) for category,sms in records]



len_of_features_set = len(features_set)



training_data_size = int(len_of_features_set*0.75)



training_set = features_set[:training_data_size]

testing_set = features_set[training_data_size: ]



print('training....')

classifier = nltk.NaiveBayesClassifier.train(training_set)



print('testing.....')

print("Accuracy: ",nltk.classify.accuracy(classifier, testing_set)*100)


