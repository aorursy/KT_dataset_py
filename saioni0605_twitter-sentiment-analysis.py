import pandas as pd

from nltk.tokenize import RegexpTokenizer

import nltk

regex=nltk.RegexpTokenizer(r'\w+')

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
df=pd.read_csv('../input/train.csv',encoding = "ISO-8859-1",low_memory=False)
df['word_count'] = df['tweet'].apply(lambda x: len(str(x).split(" "))) ## no of words

df[['tweet','word_count']].head()
df['char_count'] = df['tweet'].str.len() ## no of characters

df[['tweet','char_count']].head()
def avg_word(sentence):  

  words = sentence.split()

  return (sum(len(word) for word in words)/len(words))  ##avg word length



df['avg_word'] = df['tweet'].apply(lambda x: avg_word(x))

df[['tweet','avg_word']].head()
from nltk.corpus import stopwords    

stop = stopwords.words('english')



df['stopwords'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x in stop])) ##counting stopwords

df[['tweet','stopwords']].head()
df['hastags'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')])) ##counting special characters

df[['tweet','hastags']].head()
df['numerics'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()])) ##counting numerics

df[['tweet','numerics']].head()
## converting to lower case

df['tweet'] = df['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df['tweet'].head()
## removing punctuations

df['tweet'] = df['tweet'].str.replace('[^\w\s]','')

df['tweet'].head()
from nltk.corpus import stopwords ##removing stopwords

stop = stopwords.words('english')

df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

df['tweet'].head()
freq = pd.Series(' '.join(df['tweet']).split()).value_counts()[:10]  ##couting common word

freq
freq = list(freq.index)

df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq)) ##common word removal

df['tweet'].head()
freq = pd.Series(' '.join(df['tweet']).split()).value_counts()[-10:] ##finding rare words

freq 
freq = list(freq.index)

df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))  ## rare words removal

df['tweet'].head()
from textblob import TextBlob

df['tweet'][:5].apply(lambda x: str(TextBlob(x).correct())) ## spelling correction
TextBlob(df['tweet'][1]).words  ## Tokenization
##Lemmatization

from textblob import Word

df['tweet'] = df['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))  

df['tweet'].head()
##Lemmatization

from textblob import Word

df['tweet'] = df['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))  

df['tweet'].head()
## bage of Words

from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")

df_bow = bow.fit_transform(df['tweet'])

df_bow
## Sentiment Analysis

df['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)
df['sentiment'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )

df[['tweet','sentiment']].head()