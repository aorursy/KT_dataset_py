import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)
import numpy as np

import pandas as pd
df = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

df.head()
df['word_count'] = df['tweet'].apply(lambda x:len(str(x).split(" ")))

# df.head()

df[['tweet', 'word_count']].head()
df['char_count'] = df['tweet'].str.len() # This will include spaces / white space.

# df.head()

df[['tweet', 'char_count']].head()
def avg_word(sentence):

    words = sentence.split()

    return (sum(len(word) for word in words) / len(words))



df['avg_word'] = df['tweet'].apply(lambda x: avg_word(x))



# df.head()

df[['tweet', 'avg_word']].head()
from nltk.corpus import stopwords



stop = stopwords.words('english')
df['stopwords'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))



# df.head()

df[['tweet', 'stopwords']].head()
df['hashtags'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))



# df.head()

df[['tweet', 'hashtags']].head()
df['numerics'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))



# df.head()

df[['tweet', 'numerics']].head()
df['upper'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))



# df.head()

df[['tweet', 'upper']].head()
df['tweet_lower'] = df['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))



# df.head()

df[['tweet', 'tweet_lower']].head()
df['tweet_lower'] = df['tweet_lower'].str.replace('[^\w\s]','')

# df.head()

df['tweet_lower'].head()
df['tweet_lower'] = df['tweet_lower'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))



df.tweet_lower.head()
freq_words = pd.Series(' '.join(df['tweet_lower']).split()).value_counts()[:10]

freq_words
freq_words = list(freq_words.index)



df['tweet_lower'] = df['tweet_lower'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_words))



df['tweet_lower'].head()
rare_words = pd.Series(' '.join(df['tweet_lower']).split()).value_counts()[-10:]

rare_words
rare_words = list(rare_words.index)



df['tweet_lower'] = df['tweet_lower'].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))



df['tweet_lower'].head()
from textblob import TextBlob
df['tweet_lower'][:5].apply(lambda x: str(TextBlob(x).correct()))
TextBlob(df['tweet_lower'][1]).words 
from nltk.stem import PorterStemmer
st = PorterStemmer()
df['tweet_lower'][:5].apply(lambda x:" ".join([st.stem(word) for word in x.split()]))
from textblob import Word

df['tweet_lower'] = df['tweet_lower'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))



df.tweet_lower.head()
TextBlob(df['tweet_lower'][0]).ngrams(2)
TextBlob(df['tweet'][0]).ngrams(2)
# tf1 = (df['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

tf1 = (df['tweet_lower'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()



tf1.columns = ['words', 'tf']



tf1
for i, word in enumerate(tf1['words']):

    tf1.loc[i, 'idf'] = np.log(df.shape[0] / (len(df[df['tweet_lower'].str.contains(word)])))

    

tf1
tf1['tf_idf'] = tf1['tf'] * tf1['idf']

tf1
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(max_features = 1000, lowercase = True, analyzer = 'word', stop_words = 'english', ngram_range = (1,1))



df_vect = tfidf.fit_transform(df['tweet'])



df_vect

from sklearn.feature_extraction.text import CountVectorizer



bow = CountVectorizer(max_features=1000, lowercase = True, ngram_range= (1,1), analyzer = 'word')



df_bow = bow.fit_transform(df['tweet'])



df_bow
df['tweet'][:5].apply(lambda x: TextBlob(x).sentiment )
df['sentiment'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment[0])



df[['tweet','sentiment']].head()
df['sentiment2'] = df['tweet_lower'].apply(lambda x: TextBlob(x).sentiment[0])



df[['tweet_lower','sentiment2']].head()
from gensim.scripts.glove2word2vec import glove2word2vec