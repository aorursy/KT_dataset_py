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
from sklearn.model_selection import train_test_split



train = pd.read_csv('../input/labeledTrainData.tsv', delimiter="\t")

test = pd.read_csv('../input/testData.tsv', delimiter="\t")

test.head()  
# test data比如train data少了label的一维

print (train.shape)

print (test.shape)
'''

    清理数据，文本中包含HTML的符号比如<>，我们使用正则表达式简单地清理一下

'''

import re  #正则表达式

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer



#porter = PorterStemmer()

lancaster=LancasterStemmer()

lemmatizer = WordNetLemmatizer()

def review_preprocessing(review, stem = True, lemm = True):

    revised_words = []

    #只保留英文单词

    review_text = re.sub("[^a-zA-Z]"," ", review)

    #变成小写

    review_text = review_text.lower()    

    #Stemming or Lemmatization

    for word in word_tokenize(review_text):

        if stem:

            word = lemmatizer.lemmatize(word)

        if lemm:

            word = lancaster.stem(word)

        revised_words.append(word)

    return_words = " ".join(revised_words)

    return(return_words)



test_review = train['review'][0]

print(test_review)

return_review = review_preprocessing(test_review, stem = True, lemm = True)

print("  ")

print(return_review)
data_train, data_test = [], []

# 把训练集的文本和标注分开



# 1. 把标注提取出来

label_train = train['sentiment']

# 2. 把文本提取出来

for review in train['review']:

    data_train.append(review_preprocessing(review, stem = True, lemm = True))

    

for review in test['review']:

    data_test.append(review_preprocessing(review, stem = True, lemm = True))

# 3. 转化成numpy数组        

data_train = np.array(data_train)

data_test = np.array(data_test)



print(data_train.shape)

print(data_test.shape)
# split data to train & test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_train, label_train, test_size = 0.2, random_state = 0 )

print(x_train.shape)

print(x_test.shape)
from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.feature_extraction.text import TfidfVectorizer



# 简单的计数

#vectorize = CountVectorizer()

#data_train_count = vectorize.fit_transform(data_train)

#test_train_count = vectorize.transform(data_test)

# 使用tf-idf

tfidf = TfidfVectorizer(

           ngram_range=(1, 4),  # 二元文法模型

           use_idf=1,

           smooth_idf=1,

           stop_words = 'english') # 去掉英文停用词

data_train_count = tfidf.fit_transform(x_train)

test_train_count = tfidf.transform(x_test)



#data_train_count = tfidf.fit_transform(data_train)

#data_test_count  = tfidf.transform(data_test)
# 多项式朴素贝叶斯

from sklearn.naive_bayes import MultinomialNB 



clf_model = MultinomialNB()

clf_model.fit(data_train_count, y_train)

pred = clf_model.predict(test_train_count)

print(pred)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(pred, y_test))

print(classification_report(pred, y_test))

print(confusion_matrix(pred, y_test))
# Using final parameters of tfidf 

data_train_count = tfidf.fit_transform(data_train)

test_train_count = tfidf.transform(data_test)
clf_model = MultinomialNB()

clf_model.fit(data_train_count, label_train)

pred = clf_model.predict(test_train_count)
# 把结果保存到csv文件中，并进行提交: https://www.kaggle.com/c/word2vec-nlp-tutorial/leaderboard

df = pd.DataFrame({"id": test['id'],"sentiment": pred})

df.to_csv('submission.csv',index = False, header=True)