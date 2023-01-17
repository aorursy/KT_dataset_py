#coding:utf-8

import numpy as np

import pandas as pd

import os

import gc

import matplotlib.pyplot as plt

#import seaborn as sns



#pal=sns.color_palette()

#输入训练与测试文件

df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')

#导入停用词

from nltk.corpus import stopwords

stops=set(stopwords.words("english"))



def word_match_share(row):

    q1words= set()

    q2words=set()

    for word in str(row['question1']):

        if word not in stops:

            q1words.add(word)

    for word in str(row['question2']):

        if word not in stops:

            q2words.add(word)

    if len(q1words)==0 or len(q2words)==0:

        return 0;

    sharedwordinq1=[w for w in q1words if w in q2words]

    sharedwordinq2=[w for w in q2words if w in q1words]

    R=1.0*(len(sharedwordinq1)+len(sharedwordinq2))/(len(q1words)+len(q2words))

    return R



#特征1，共有词除于总长度

train_word_match=df_train.apply(word_match_share,axis=1,raw=True)

test_word_match=df_test.apply(word_match_share,axis=1,raw=True)

common_train=pd.DataFrame()

common_train['train_id']=df_train['id']

common_train['common_word']=train_word_match

common_train['is_duplicate']=df_train['is_duplicate']

common_train.to_csv('train_common_word.csv',index=False)

common_test=pd.DataFrame()

common_test['test_id']=df_test['test_id']

common_test['common_word']=test_word_match

common_test.to_csv('test_common_word.csv',index=False)