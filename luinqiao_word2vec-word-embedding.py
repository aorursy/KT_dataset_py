# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import re

import json

import jieba

jieba.suggest_freq("##", tune=True)

corpus_path = '../input/news-corpus/news2016zh_train.json'

train_path = '../input/newssentiment/Train_DataSet.csv'

test_path = '../input/newssentiment/Test_DataSet.csv'



def filters(text):

    try:

        number = re.compile(r"[0-9]{1,}")

        text = number.sub('##', text)

        english = re.compile(r"[a-zA-Z]{1,}")

        text = english.sub('', text)

        non_chinese = re.compile('[^\u4E00-\u9FA5]{6,}')

        text = non_chinese.sub('', text)

        text = ' '.join(jieba.cut(text)).split()

    except:

        return []

    return text



def internal_data():

    data = []

    train = pd.read_csv(train_path)

    test= pd.read_csv(test_path)

    data.extend(train['content'].values)

    data.extend(train['title'].values)

    data.extend(test['content'].values)

    data.extend(test['title'].values)

    for i, line in enumerate(data):

        data[i] = filters(line)

    return data





class iterator(object):

    def __init__(self):

        self.internal = internal_data()

    def __iter__(self):

        fr = open(corpus_path, 'r', encoding='utf8')

        for i, line in enumerate(fr):

            line = json.loads(line)['content']

            line = filters(line)

            if i > 600000:

                break

            yield line

        fr.close()

        for line in self.internal:

            yield line

            
corpus = iterator()

model = Word2Vec(corpus, size=300, window=5, min_count=0, workers=5)

model.save('word2vec.model')