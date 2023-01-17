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
import nltk
import random
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
import string
import pandas as pd
df = pd.read_csv("./../input/spam.csv", encoding = 'latin-1')
print(df.head())  
message_list = []                                                        
word_list = []                                                           

for index, row in df.iterrows(): 
    
    #print(row['v1'], row['v2'])
    
    category = row['v1']
    message  = row['v2']
    
    message_list.append([message, category])
    
    for s in string.punctuation:                                        
        if s in message:
            message = message.replace(s, " ")
               
    stop = stopwords.words('english')
    for word in message.split(" "):                                        
        if not word in stop:
            word_list.append(word.lower())
message_list[0:5]
random.shuffle(message_list)
word_list = nltk.FreqDist(word_list)
print("words len : ", len(list(word_list.keys())))
word_features = list(word_list.keys())
def find_feature(word_features, message):
    feature = {}
    for word in word_features:
        feature[word] = word in message.lower()
    return feature
featureset = [(find_feature(word_features, message), category) for (message, category) in message_list]
trainingset = featureset[:int(len(featureset)*3/4)]
testingset = featureset[int(len(featureset)*3/4):]
len(featureset)
len(trainingset)
len(testingset)
NBC = nltk.NaiveBayesClassifier.train(trainingset)
nbc_accuracy = nltk.classify.accuracy(NBC, testingset)*100
nbc_accuracy
NBC.show_most_informative_features(10)
for message, category in message_list[0:10]:
    feature = find_feature(word_features, message)
    print(message, "-->", NBC.classify(feature))
from sklearn.linear_model import SGDClassifier

SGDC = SklearnClassifier(SGDClassifier())
SGDC.train(trainingset)
feature = find_feature(word_features, message)
print(SGDC.classify(feature))
print("SGD Classifier accuracy = " + str((nltk.classify.accuracy(SGDC, testingset))*100))
from sklearn.linear_model import LogisticRegression
# Logistic Regression classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(trainingset)
print("LRC accuracy: "+ str((nltk.classify.accuracy(LogisticRegression_classifier, testingset))*100))