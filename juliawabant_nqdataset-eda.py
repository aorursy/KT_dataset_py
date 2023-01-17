import os

import pandas as pd

import json

import matplotlib.pyplot as plt

import seaborn as sns

import dask.bag as db



from tf_qa_jsonl_to_dataframe import jsonl_to_df

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.max_colwidth', -1)



sns.set(rc={'figure.figsize':(11.7,8.27)})

plt.style.use('seaborn-darkgrid')
os.listdir("../input")
train_preview = []



with open('../input/tensorflow2-question-answering/simplified-nq-train.jsonl', 'rt') as big_file:

    for i in range(10000):

        train_preview.append(json.loads(big_file.readline()))
train_preview_df = pd.DataFrame(train_preview)
train_preview_df.head()
train_preview_df.tail()
train_preview_df.info()
y = train_preview_df['annotations'][0][0]
print('Example question : ', train_preview_df['question_text'][0])

print('Example short answer tokens indices:', y['short_answers'][0]['start_token'], y['short_answers'][0]['end_token'])

print('Example long answer tokens indices: ', y['long_answer']['start_token'], y['long_answer']['end_token'])
directory = '/kaggle/input/tensorflow2-question-answering/'

train = jsonl_to_df(directory + 'simplified-nq-train.jsonl', n_rows=-1)

test = jsonl_to_df(directory + 'simplified-nq-test.jsonl', load_annotations=False)
train.head()
print(len(train), len(test))
yn_answers = train['yes_no_answer']
pd.Series(yn_answers).value_counts().sort_index().plot(kind='bar')
pd.Series(yn_answers).value_counts()
train['long_answer_start'].describe()
train['long_answer_end'].describe()
print(len(train['long_answer_start'] < 0), len(train['long_answer_end'] < 0))
tags =  pd.read_csv('../input/tags-html/html_tags.txt', header=None)

tags
train['document_text_ntags'] = train['document_text'].str.lower()

train['document_text_ntags'] = train['document_text_ntags'].str.split(' ').replace([i for i in tags], '')
train['document_text_ntags'] = [BeautifulSoup(text,"lxml").get_text() for text in train['document_text']]
train['document_text_ntags'].head()
from yellowbrick.text import FreqDistVisualizer
vectorizer = CountVectorizer()

docs       = vectorizer.fit_transform(train['document_text_ntags'])

features   = vectorizer.get_feature_names()



visualizer = FreqDistVisualizer(features=features, orient='v')

visualizer.fit(docs)

visualizer.show()