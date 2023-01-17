from bs4 import BeautifulSoup    
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库

import pandas as pd


train = pd.read_csv('../input/labeledTrainData.tsv', delimiter="\t")
test = pd.read_csv('../input/testData.tsv', delimiter="\t")
train.head()    
train.info()
# test data比如train data少了label的一维
print (train.shape)
print (test.shape)
print(train.loc[0]['review'])
'''
    清理数据，文本中包含HTML的符号比如<>，我们使用正则表达式简单地清理一下
'''
import re  #正则表达式

def review_to_wordlist(review):
    #只保留英文单词
    review_text = re.sub("[^a-zA-Z]"," ", review)
    
    #变成小写
    words = review_text.lower()
    
    return(words)

# 把训练集的文本和标注分开
# 1. 把标注提取出来
y_train = train['sentiment']

# 2. 把文本提取出来
train_data = []
for review in train['review']:
    review_no_markup = BeautifulSoup(review, "lxml").get_text()
    train_data.append(review_to_wordlist(review_no_markup))

# 3. 转化成numpy数组        
train_data = np.array(train_data)

# 对测试集的文本做同样的事情
test_data = []
for review in test['review']:
    review_no_markup = BeautifulSoup(review, "lxml").get_text()
    test_data.append(review_to_wordlist(review_no_markup))
    
test_data = np.array(test_data)

print(train_data.shape)
print(test_data.shape)
print(train_data[0])
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

# 简单的计数
#vectorizer = CountVectorizer()
#data_train_count = vectorizer.fit_transform(train_data)
#data_test_count  = vectorizer.transform(test_data)

# 使用tf-idf
tfidf = TfidfVectorizer(
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词


data_train_count = tfidf.fit_transform(train_data)
data_test_count  = tfidf.transform(test_data)

print("Let's go!")
# 多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB 

clf = MultinomialNB()
clf.fit(data_train_count, y_train)
pred = clf.predict(data_test_count)
print (pred)
# 把结果保存到csv文件中，并进行提交: https://www.kaggle.com/c/word2vec-nlp-tutorial/leaderboard
df = pd.DataFrame({"id": test['id'],"sentiment": pred})

df.to_csv('submission.csv',index = False, header=True)




