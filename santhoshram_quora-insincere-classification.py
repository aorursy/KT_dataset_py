# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings



warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train_df = train.copy()

test_df = test.copy()
(train.columns,test.columns)
train.sample()

test.head()
target_counts = train['target'].value_counts()

print("Sincere questions:%i (%.1f%%)" %(target_counts[0],float(100*target_counts[0]/(target_counts[0]+target_counts[1]))))

print("Insincere questions: %i (%.1f%%)" %(target_counts[1],float(100*target_counts[1]/(target_counts[0]+target_counts[1]))))
sns.countplot(train['target'], hue=train["target"])

plt.show()
train['target'].value_counts().plot.bar()

plt.show()
fig1, ax1 = plt.subplots()

ax1.pie(train['target'].value_counts().values, explode=(0,0.1),labels=['Sincere','Insincere'], autopct='%1.1f%%',

        shadow=True, startangle=130)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
train['target'].value_counts().values
"""

#Example for Ngraming using nltk

from nltk import ngrams



sentence = 'I am Dr. Santhosh'



n = 2

sixgrams = ngrams(sentence.split(), n)



for grams in sixgrams:

  print(grams)

"""
"""

#Example For Ngraming method without using nltk

#sentence = 'this is a foo bar sentences and i want to ngramize it'

sentence = "I am Dr. Santhosh"

tokens = sentence.split(" ")

sequence = [tokens[i:] for i in range(3)]

onegram = zip(*sequence)

#print(sequence)

#print(onegram)

gramList = [" ".join(ngram) for ngram in onegram]

print(gramList)

"""
from nltk import ngrams

from wordcloud import WordCloud, STOPWORDS

from collections import defaultdict



sincere_text = train[train['target']==1]

insincere_text = train[train['target']==0]

def generate_ngrams(sentence, n):

    tokens = [token for token in sentence.lower().split(" ") if token != "" if token not in STOPWORDS]

    return [" ".join(grams) for grams in ngrams(tokens, n)]

def get_ngram_count(df,n):

    freq_dict = defaultdict(int)

    for text in df["question_text"]:

        for word in generate_ngrams(text,n):

            freq_dict[word] += 1

            

    return freq_dict

print(get_ngram_count(insincere_text,3))

#generate_ngrams("I am Dr. Santhosh", 2)