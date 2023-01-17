# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.feature_extraction.text import TfidfVectorizer

import time

from sklearn import svm

from sklearn.metrics import classification_report

import pandas as pd

from decimal import *

from nltk.stem.porter import PorterStemmer

import re

from stop_words import get_stop_words

import itertools

import numpy as np

from sklearn import metrics
trainData = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

trainData.sample(6)

testData=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

testData.sample(6)

sub=pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
def del_miss_val(df):

    total=df.isnull().sum()

    return pd.concat([total],axis=1,keys=['Total'])

print("Missing values for train dataset \n")

print(del_miss_val(trainData))

trainData=trainData.dropna()

print(len(trainData["sentiment"]))

print(len(trainData["text"]))
def remove_punct(text):

    line = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]+'," ",text)

    return " ".join(line.split())

v=remove_punct('Sons of ****, why couldn`t they put them on t...')

v
def delete_link(string): 

    url = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', string)

    return "".join(url) 

test=delete_link('last session of the day http://twitpic.com/67ezh')

test
train= [delete_link(str(text)) for text in trainData['text']]

# train = pd.DataFrame(train)

# train
train= [remove_punct(str(text)) for text in train]

train = pd.DataFrame(train)

train
test= [delete_link(text) for text in testData['text']]

testData['text']= [remove_punct(str(text)) for text in test]

testData['text'] = pd.DataFrame(testData['text'])

testData['text']
def remove_stpwd(text,langue='en'):

    text=text.split()

    stop_words = get_stop_words(langue)

    text=' '.join([token for token in text if token not in stop_words and len(token)>1])

    return text
x=remove_stpwd('Shanghai is also really exciting precisely sky')

x
porter_stemmer=PorterStemmer()

words=testData['text']

words = [porter_stemmer.stem(word) for word in words]



words=[remove_stpwd(word) for word in words]

# test['selected_text'] = stemming_tokenizer()

# submission=testData[['textID','selected_text']]

# submission.to_csv('submission.csv',index=False)

# submission.head(5)

words
testData['selected_text'] = words

submission=testData[['textID','selected_text']]

submission.to_csv('submission.csv',index=False)

submission.head(5)