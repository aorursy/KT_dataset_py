# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", encoding="latin-1")

train.head()

train["word_count"]=train['SentimentText'].apply(lambda x: len(str(x).split(" ")))

train[['word_count', 'SentimentText']].head(5)
train["character_count"]=train["SentimentText"].str.len()

train[["character_count", "SentimentText"]].head(5)
def avg_word(sentence):

    words=sentence.split()

    return(sum(len(word) for word in words)/len(words))



train["word_length"]=train["SentimentText"].apply(lambda x: avg_word(x))

train[["word_length", "SentimentText"]].head()

    
from nltk.corpus import stopwords

stop = stopwords.words('english')



train["stop_words"]=train["SentimentText"].apply(lambda x: len([x for x in x.split() if x in stop]))

train[["stop_words", "SentimentText"]].head()

                                                 
train["hashtags"]=train["SentimentText"].apply(lambda x: len([x for x in x.split() if x.startswith("#")]))

train[["hashtags", "SentimentText"]].head(10)        
train["numerics"]=train["SentimentText"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

train[["numerics", "SentimentText"]].head()
train["upper"]=train["SentimentText"].apply(lambda x: len([x for x in x.split() if x.isupper()]))

train[["upper", "SentimentText"]].head()
train['SentimentText'] = train['SentimentText'].apply(lambda x: (" ".join(x.lower() for x in x.split())))

train['SentimentText'].head(10)

train["SentimentText"]=train["SentimentText"].str.replace('[^\w\s]', '')

train["SentimentText"].head(10)
from nltk.corpus import stopwords

stop=stopwords.words('english')



train["SentimentText"]=train["SentimentText"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

train["SentimentText"].head(10)
freq=pd.Series(' '.join(train["SentimentText"]).split()).value_counts()[:20]

freq
import matplotlib.pyplot as plot

freq.plot(kind="bar")
freq=list(freq.index)

freq
train["SentimentText"]=train["SentimentText"].apply(lambda x: " ".join(x for x in x.split()if x not in freq))

train["SentimentText"].head(10)
freq=pd.Series(" ".join(train["SentimentText"]).split()).value_counts()[-10:]

freq
train["SentimentText"]=train["SentimentText"].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

train["SentimentText"].head(10)
from textblob import TextBlob

train["SentimentText"][:5].apply(lambda x: str(TextBlob(x).correct()))
TextBlob(train["SentimentText"][3]).words
from textblob import Word

train["SentimentText"]=train["SentimentText"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

train["SentimentText"].head(7)
TextBlob(train["SentimentText"][1]).ngrams(2)