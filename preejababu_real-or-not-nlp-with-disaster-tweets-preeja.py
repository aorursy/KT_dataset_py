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
import numpy as np

import pandas as pd 

import seaborn as sns

import re

import nltk

from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

from nltk.util import ngrams

from nltk.stem.porter import PorterStemmer

from collections import  Counter

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from nltk.tokenize import word_tokenize

import gensim

import string

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from tensorflow.keras.initializers import Constant

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam

import warnings

from collections import defaultdict

warnings.filterwarnings('ignore')

print("Important libraries loaded successfully")
#Importing and understanding the structure of  Data

data_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

print("Train Data shape = ",data_train.shape)

data_train.head(2)
data_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

print("Test Data shape = ",data_test.shape)

data_test.head(2)
missing_cols = ['id','keyword', 'location','text']



fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)



sns.barplot(x=data_train[missing_cols].isnull().sum().index, y=data_train[missing_cols].isnull().sum().values, ax=axes[0])

sns.barplot(x=data_test[missing_cols].isnull().sum().index, y=data_test[missing_cols].isnull().sum().values, ax=axes[1])



axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)

axes[0].tick_params(axis='x', labelsize=15)

axes[0].tick_params(axis='y', labelsize=15)

axes[1].tick_params(axis='x', labelsize=15)

axes[1].tick_params(axis='y', labelsize=15)



axes[0].set_title('Training Set', fontsize=13)

axes[1].set_title('Test Set', fontsize=13)



plt.show()



for df in [data_train, data_test]:

    for col in ['keyword', 'location']:

        df[col] = df[col].fillna(f'no_{col}')
x=data_train.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

data_train_len=data_train[data_train['target']==1]['text'].str.len()

ax1.hist(data_train_len,color='red')

ax1.set_title('disaster tweets')

data_train_len=data_train[data_train['target']==0]['text'].str.len()

ax2.hist(data_train_len,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=data_train[data_train['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='red')

ax1.set_title('disaster tweets')

tweet_len=data_train[data_train['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Words in a tweet')

plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

word=data_train[data_train['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')

ax1.set_title('disaster')

word=data_train[data_train['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')

ax2.set_title('Not disaster')

fig.suptitle('Average word length in each tweet')
def create_corpus(target):

    corpus=[]

    # Split the tweet text into words and append into corpus list 

    for x in data_train[data_train['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
# list of corpus for type 0 tweet 

corpus=create_corpus(0)

#Frequency calculation of stop words in corpus list of type 0 tweet 

dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

#Sorting of frequency values and displaying top ten frequencies        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)

plt.bar(x,y)
# list of corpus for type 1 tweet 

corpus=create_corpus(1)

#Frequency calculation of stop words in corpus list of type 0 tweet 

dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

#Sorting of frequency values and displaying top ten frequencies        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)

plt.bar(x,y)