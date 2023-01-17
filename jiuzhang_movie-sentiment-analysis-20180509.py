import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库

import pandas as pd

train = pd.read_csv("../inputs/labeledTrainData.tsv", delimiter = '\t')


test = pd.read_csv("../inputs/testData.tsv", delimiter = '\t')
# test.head()
train.head()
print (train.shape, test.shape)
train.review
import re

# 去掉标点
def review_to_wordlist(review):
    review_text = re.sub("[^a-zA-Z]", " ", review)
    
    
    return review_text.lower()

train_data =[]
for document in train.review.values:
    res = review_to_wordlist(document)
    train_data.append(res)
#     print(document)
#     print('\n\n')
#     print(res)
    


test_data =[]
for document in test.review.values:
    res = review_to_wordlist(document)
    test_data.append(res) 
        
test_data
from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer()

data_train_count = vectorizer.fit_transform(train_data) # fit 统计单词， transform 把每句话转变为词向量
data_test_count = vectorizer.transform(test_data)
print(data_train_count.shape, data_test_count.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tfidf = TfidfVectorizer()

data_train_tfidf = vectorizer_tfidf.fit_transform(train_data) # fit 统计单词， transform 把每句话转变为词向量
data_test_tfidf = vectorizer_tfidf.transform(test_data)
print(data_train_tfidf[0])
print (data_train_count[0])
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
y_train = train.sentiment
clf.fit(data_train_tfidf, y_train) # fit train的过程
predictions = clf.predict(data_test_tfidf) # 预测过程
print(predictions)
# print (data_train_tfidf[0])

