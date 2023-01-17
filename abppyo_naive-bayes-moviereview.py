import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



train = pd.read_csv('../input/labeledTrainData.tsv', delimiter = '\t') #delimiter 以空格分词

test = pd.read_csv('../input/testData.tsv',delimiter = '\t')



train.head()



test.head()


#test data比train data 少label一列

print(train.shape)

print(test.shape)
#清理数据 文本中包含HTML符号比如<> 用正则表达式清理



import re #正则表达式



def review_preprocessing(review):

    #只保留英文单词

    review_text = re.sub("[^a-zA-Z]"," ", review)

    

    #变成小写

    words = review_text.lower()

    return words



#把训练集文本和标注分开

#1.提取标注

y_train = train['sentiment']



#2.提取文本

train_data = []

for review in train['review']:

    train_data.append(review_preprocessing(review))



#3.化成numpy数组

train_data = np.array(train_data)



#对测试文本重复以上操作

test_data = []

for review in test['review']:

    test_data.append(review_preprocessing(review))

test_data = np.array(test_data)



print(test_data.shape)

print(train_data.shape)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



# 简单的计数

# vectorizer = CountVectorizer()

# data_train_count = vectorizer.fit_transform(train_data)

# data_test_count  = vectorizer.transform(test_data)



# 使用tf-idf

tfidf = TfidfVectorizer(

           ngram_range = (1,3), #二元文法模型

           use_idf = 1,

           smooth_idf = 1,

           stop_words = 'english') #去掉英文停用词

data_train_count = tfidf.fit_transform(train_data)

data_test_count = tfidf.transform(test_data)



#多项式朴素贝叶斯

from sklearn.naive_bayes import MultinomialNB



clf = MultinomialNB()

clf.fit(data_train_count, y_train)

pred = clf.predict(data_test_count)

print(pred)
#保存csv

df = pd.DataFrame({"id":test['id'],"sentiment":pred})

df.to_csv('submission.csv',index=False,header=True)