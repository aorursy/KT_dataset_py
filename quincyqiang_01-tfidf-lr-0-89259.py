import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.head()
import re

import string

import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer



# 加载数据

stop_words = set(stopwords.words('english'))

# words=[word for word in word_tokenize(data) if word not in stop_words]



# 数据清洗

text_pattern = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')



def tokenize(s):

    return word_tokenize(text_pattern.sub(' \1', s))





vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize, min_df=3,

                      max_df=0.9, stop_words=stop_words,

                      strip_accents='unicode', use_idf=1,

                      smooth_idf=1, sublinear_tf=1)

print("getting tfidf vec....")

x_train = vec.fit_transform(train["comment_text"])

x_test = vec.transform(test["comment_text"])

y = np.where(train['target'] >= 0.5, 1, 0)

print("done getting tfidf vec....")
# #!/usr/bin/env python  

# # -*- coding:utf-8 _*-  

""" 

@author:quincyqiang 

@license: Apache Licence 

@file: 01_lr.py 

@time: 2019-04-21 14:37

@description:

"""

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



print("traing lr...")

clf=LogisticRegression(C=4,dual=True)

clf.fit(x_train,y)

prediction=clf.predict_proba(x_test)[:,1]

print(prediction)

submission = pd.read_csv("../input/sample_submission.csv")

submission['prediction'] = prediction

submission.to_csv('submission.csv', index=False)
