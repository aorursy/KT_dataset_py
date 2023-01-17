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
#import all the libraries needed

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import re  

import nltk

import string

from bs4 import BeautifulSoup

from nltk import sent_tokenize, word_tokenize

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from textblob import TextBlob
df = pd.read_csv('/kaggle/input/asos-customer-review-in-trustpilot/asos_transform.csv')
#view example

df['Contents'][4]
def remove_punctuation(text):

    no_punct="".join([c for c in text if c not in string.punctuation])

    return no_punct
df['Contents'] = df['Contents'].apply(lambda x: remove_punctuation(x))

df['Contents'].head()
tokenizer = RegexpTokenizer(r'\w+')
df['Contents']=df['Contents'].apply(lambda x: tokenizer.tokenize(x.lower()))

df['Contents'].head()
def remove_stopwords(text):

    words = [w for w in text if w not in stopwords.words('english')]

    return words
df['Contents']= df['Contents'].apply(lambda x : remove_stopwords(x))

df['Contents'].head(10)
#Instatiate Stemmer

stemmer = SnowballStemmer('english')

def word_stemmer(text):

    stem_text = " ".join([stemmer.stem(i) for i in text])

    return stem_text
df['Contents']=df['Contents'].apply(lambda x:word_stemmer(x))

df['Contents'].head(10)
df_results = df['Contents'].apply(lambda x: TextBlob(x).sentiment)

print(df_results)
bloblist_tags = list()



df_str =df['Contents']

for row in df_str:

    blob = TextBlob(row)

    bloblist_tags.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))

    df_str = pd.DataFrame(bloblist_tags, columns = ['sentence','sentiment','polarity'])



def f_tags(df_str):

    if df_str['sentiment'] > 0:

        val = "Positive"

    elif df_str['sentiment'] == 0:

        val = "Neutral"

    else:

        val = "Negative"

    return val



df_str['Sentiment_Type'] = df_str.apply(f_tags, axis=1)



plt.figure(figsize=(10,10))

sns.set_style("whitegrid")

ax = sns.countplot(x="Sentiment_Type", data=df_str)