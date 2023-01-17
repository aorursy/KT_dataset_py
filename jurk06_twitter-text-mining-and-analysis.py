import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
data=pd.read_csv('../input/data.csv', sep='\t',quoting=3)
data.head()
cols=['sentiment', 'text', 'user']
df=pd.read_csv('../input/data.csv', names=cols)
df.head()
df.drop(df.index[0], inplace=True)
df.head()
def remove_puntuation(text):
    import string
    transtor=str.maketrans('', '', string.punctuation)
    return text.translate(transtor)
df['text']=df['text'].apply(remove_puntuation)
df.head()
df['length_text']=[len(t) for t in df.text]
df.head()
sw=stopwords.words('english')
np.array(sw)
print("number of stopwords: ", len(sw))
def stop_words(text):
    text=[word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)
df['text']=df['text'].apply(stop_words)
df.head()
len(df['text'].iloc[0])
df.drop(['user', 'length_text'], axis=1, inplace=True)
df.head()
df['length_text']=[len(t) for t in df.text]
df.head()
count_vecorize=CountVectorizer()
count_vecorize.fit(df['text'])
dictionary=count_vecorize.vocabulary_.items()
vocab=[]
count=[]

for key, val in dictionary:
    vocab.append(key)
    count.append(val)
    
vocab_bef_stem=pd.Series(count, index=vocab)
vocab_bef_stem=vocab_bef_stem.sort_values(ascending=False)
top_vocabs=vocab_bef_stem.head(20)
top_vocabs.plot(kind='barh', figsize=(8,10), xlim=(10,750))
stemmer=SnowballStemmer('english')
def stemming(text):
    text=[stemmer.stem(word) for word in text.split()]
    return " ".join(text)
df['text']=df['text'].apply(stemming)
df.head()
tfid_vectorize=TfidfVectorizer('english')
tfid_vectorize.fit(df['text'])
dictionary=tfid_vectorize.vocabulary_.items()

vocab=[]
count=[]

for key, val in dictionary:
    vocab.append(key)
    count.append(val)
    
vocab_after_stem=pd.Series(count, index=vocab)

vocab_after_stem=vocab_after_stem.sort_values(ascending=False)

top_vocabs=vocab_after_stem.head(20)
top_vocabs.plot(kind='barh', figsize=(8,15), xlim=(200, 750))
df.head()
df.drop(['length_text'], axis=1, inplace=True)
df.head()
df['length_text']=[len(t) for t in df.text]
df.head()
N_data=df[df['sentiment']=='neutral']
P_data=df[df['sentiment']=='positive']
Ne_data=df[df['sentiment']=='negative']

plt.rcParams['figure.figsize']=(12,6)
bins=200
plt.hist(N_data['length_text'], alpha=0.6, bins=bins, label='Neutral')
plt.hist(P_data['length_text'], alpha=0.6, bins=bins, label='Positive')
plt.hist(Ne_data['length_text'], alpha=0.6, bins=bins, label='Negative')

plt.legend()

plt.show()
