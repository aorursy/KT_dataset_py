import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import os

import warnings

warnings.filterwarnings("ignore")

import os

for file in os.listdir('../input/'):

    print(file.ljust(30)+str(round(os.path.getsize('../input/'+file)/1000000,2))+'MB')
ckpt = datetime.now()

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv',index_col='test_id')

print(f'time cost: {datetime.now() - ckpt}')
print(len(train_data[train_data['is_duplicate']==1]))

print(len(train_data[train_data['is_duplicate']==0]))

print('The ratio:',len(train_data[train_data['is_duplicate']==1])/len(train_data))
# from sklearn.metrics import log_loss

# probability = train_data['is_duplicate'].mean()

# naive_loss = log_loss(train_data['is_duplicate'].astype('float'),np.zeros((len(train_data),1))+probability)

# print(naive_loss)

# print(train_data[train_data['is_duplicate']==1][['question1', 'question2']].head(5))

# print(train_data[train_data['is_duplicate']==0][['question1', 'question2']].head(5))
import psutil

psutil.virtual_memory()
print("Let's check Nan before processing for train data")

from IPython.display import display, HTML

print('In training data')

print('\nquestion1 is nan:')

display(train_data[train_data['question1'].apply(lambda x:x is np.nan)])

print('question2 is nan:')

display(train_data[train_data['question2'].apply(lambda x:x is np.nan)])





print("Let's check Nan before processing for test data")

print('In testing data')

print('\nquestion1 is nan:')

display(test_data[test_data['question1'].apply(lambda x:x is np.nan)])

print('question2 is nan:')

display(test_data[test_data['question2'].apply(lambda x:x is np.nan)])



print('special case')

display(test_data[(test_data['question1'].apply(lambda x:x is np.nan)) & (test_data['question2'].apply(lambda x:x is np.nan))])

# test_data.loc[['life in dublin?"'],:]



print('Give 0 to these test rows in default.')
psutil.virtual_memory()
corpus =list(train_data['question1'])+list(train_data['question2'])

print('the nan: ',list(filter(lambda x:x[0] if x[1] is np.nan else None, enumerate(corpus))))

corpus = list(map(lambda x:'' if x is np.nan else x, corpus))
import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

from gensim.models import FastText

corpus_tokenized = [nltk.word_tokenize(str(sent)) for sent in corpus]

word_embedding_model = FastText(corpus_tokenized,window=5, min_count=0,workers=1000)

word_embedding_model.save('./fasttext.model')
def embedding_after_filter_stopwords():

    for idx,sent in enumerate(corpus):

        corpus[idx] = list(filter(lambda word: word not in stop_words,sent))

    word_embedding_model = FastText(corpus,window=5, min_count=1)

    word_embedding_model.save('./fasttext_with_stopwords.model')

    return word_embedding_model
from sklearn.feature_extraction.text import TfidfVectorizer

ckpt = datetime.now()

vectorizer = TfidfVectorizer()

vectorizer.fit(corpus)

print(f'Time cost: {datetime.now()-ckpt}')
train_data[train_data['question1'].apply(lambda x:x is np.nan)]
filter_npnan(train_data.loc[363362,'question1'])
ckpt = datetime.now()

def filter_npnan(x):

    return '' if x is np.nan else x

x1 = train_data['question1'].apply(filter_npnan)

x2 = train_data['question2'].apply(filter_npnan)

print('Filtered')

x1 = vectorizer.transform(x1)

x2 = vectorizer.transform(x2)

print(f'Time cost: {datetime.now()-ckpt}')

print(type(x1))

print(type(x2))
y = train_data['is_duplicate']
print('The size of bag of words after tf-ifd',len(vectorizer.get_feature_names()))

print('corpus length:',len(corpus))

print('Shape:',x1.shape)

print('Shape:',x2.shape)
print('type of tfidf vocab',type(vectorizer.vocabulary_))

print('length of the vocab',len(vectorizer.vocabulary_))
from scipy.sparse import hstack

x = hstack([x1,x2])
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
est = RandomForestRegressor()

est.fit(x_train,y_train)

print(est.score(x_test, y_test))
# x_train = np.random.randn(100,6)

# y_train = x_train[:,0]*5+x_train[:,1]*2+x_train[:,3]*4
from catboost import Pool, CatBoostRegressor, cv

from catboost import CatBoostClassifier

model = CatBoostClassifier(

    iterations=5,

    random_seed=0,

    learning_rate=0.1

)

model.fit(

   x_train, y_train,

    logging_level='Verbose',

    plot=True

);
import tensorflow as tf