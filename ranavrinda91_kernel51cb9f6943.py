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
import pandas as pd

train_df =pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')

insincere=train_df[['qid','question_text']].where(train_df['target']==1 & ~pd.isnull(train_df['question_text']))

insincere =insincere.dropna()

print (f"Total questions {train_df.shape[0]}")

print(f"Total insincere questions {insincere.shape[0]}")

insincere.head()

import nltk

import re

from nltk.corpus import stopwords

import string

from nltk.tokenize import sent_tokenize, word_tokenize   

from nltk.tokenize import WordPunctTokenizer

import gensim

from nltk.stem import WordNetLemmatizer, SnowballStemmer



punc = WordPunctTokenizer()

lemmatizer = WordNetLemmatizer() 

stop = stopwords.words('english')

exclude = set(string.punctuation)



def clean_text(text):

    word_tokens = (word_tokenize(text))

    remove_stop=[w.lower() for w in word_tokens if w.lower() not in stop]

    remove_punct=[c for c in remove_stop if c not in exclude and len(c)>3]

    clean =" ".join([re.sub(r'[^a-zA-Z0-9]','',i) for i in remove_punct ])

    lemma=lemmatizer.lemmatize(clean)

    return lemma



insincere['clean_question']=insincere['question_text'].map(clean_text)

insincere['clean_question']
# lets find out the total no of words , bigrams and trigrams

word_freq={}

for w in insincere['clean_question'].values.tolist():

    for word in w.split(" "):

        if word  in word_freq.keys():

            word_freq[word]+=1

        else:

            word_freq[word]=1

print (len(word_freq))

lists = sorted(word_freq.items())





words_df=pd.DataFrame(lists,columns=['words','frequency']).sort_values(by='frequency',ascending=False)

words_df.set_index('words')[:50].plot(kind='bar',figsize=(20,10),title='Frequency Dist For Top 20 Words');
#lets find out the most frequent bigrams

import nltk

from nltk.util import ngrams

from nltk.collocations import BigramCollocationFinder

w_list=insincere['clean_question'].map(lambda text:word_tokenize(text))

bigm_freq_dict={}

bigrams= (nltk.bigrams(w) for w in w_list)





    
bigr_freq_list=[]

bigr_freq_dict={}

def get_bigrams():

    for bi in bigrams:

        yield bi

big_obj=get_bigrams()   

for bigr in big_obj:

    bigrams  = next(bigr)

    if bigrams not in bigr_freq_list:

        bigr_freq_dict[bigrams]=0

    else:

        bigr_freq_dict[bigrams]+=1

#for i in w_list:

 #   word_fd = nltk.FreqDist(i)

 #   bigram_fd = nltk.FreqDist(nltk.bigrams(i))

 #   bigr_freq_list.extend(bigram_fd.most_common())
