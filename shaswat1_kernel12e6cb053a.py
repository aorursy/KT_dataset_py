# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import nltk

import seaborn as sns

import re

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

stop=set(stopwords.words('english'))

from collections import  Counter

from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D, Bidirectional, LeakyReLU, Dropout

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")

sub = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
plt.rcParams['patch.force_edgecolor'] = True

plt.rcParams['figure.figsize'] = (12,7)
nltk.download('stopwords')
train.head()
print('Number of rows in train dataset are: {} \nNumber of columns in train dataset are: {} '

      .format(train.shape[0],train.shape[1]))
train['target'].value_counts().plot(kind = 'bar')

plt.xlabel('Targets')

plt.ylabel('Count of Targets')

plt.xticks(rotation=0)

plt.plot()
# determining character in tweets

fig,(ax1,ax2) = plt.subplots(1,2)

tweets = train[train['target'] ==1]['text'].str.len()

ax1.hist(tweets)

ax1.set_title('Disaster Tweets')

tweets = train[train['target']==0]['text'].str.len()

ax2.hist(tweets)

ax2.set_title('Non-Disaster Tweets')

plt.suptitle('Character in Tweets')

plt.show()
# Disaster Tweets tend to be a litter shorter than non-disaster tweets
# Checking number of words in a tweet



fig,(ax1,ax2) = plt.subplots(1,2)

tweet_len = train[train['target']==1]['text'].str.split().map(lambda x:len(x))

ax1.set_title('Disaster Tweets')

ax1.hist(tweet_len)

tweet_len = train[train['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len)

ax2.set_title('Non-Disaster Tweets')

plt.suptitle('Words in Tweets')

plt.show()
def create_corpus(target):

    corpus=[]

    

    for x in train[train['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
# First we will analyze tweets with class 0.

corpus=create_corpus(0)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)

plt.bar(x,y)
# Now,we will analyze tweets with class 1

corpus=create_corpus(1)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1



top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 


x,y=zip(*top)

plt.bar(x,y)
# analysing punctuation for class 0

corpus=create_corpus(0)



dic=defaultdict(int)

import string

special = string.punctuation

for i in (corpus):

    if i in special:

        dic[i]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y)
# analysing punctuation for class 1

corpus=create_corpus(1)



dic=defaultdict(int)

import string

special = string.punctuation

for i in (corpus):

    if i in special:

        dic[i]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y)
# N-gram analysis

def get_top_tweet_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10,5))

top_tweet_bigrams=get_top_tweet_bigrams(train['text'])[:10]

x,y=zip(*top_tweet_bigrams)

plt.bar(x,y)
#Cleaning dataset
df=pd.concat([train,test])

df.shape
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
df['text']=df['text'].apply(lambda x : remove_URL(x))
#removing html tags
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)
df['text']=df['text'].apply(lambda x : remove_html(x))
#Removing emojis
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
df['text']=df['text'].apply(lambda x: remove_emoji(x))
#Removing punctuations
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)
#Correcting spellings
# spell = SpellChecker()

# def correct_spellings(text):

#     corrected_text = []

#     misspelled_words = spell.unknown(text.split())

#     for word in text.split():

#         if word in misspelled_words:

#             corrected_text.append(spell.correction(word))

#         else:

#             corrected_text.append(word)

#     return " ".join(corrected_text)
# df['text']=df['text'].apply(lambda x : correct_spellings(x))
# 100d Gove Vecorisation
def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus
corpus=create_corpus(df)
embedding_dict={}

with open('/kaggle/input/glove6b200d/glove.6B.200d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN=50

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,200))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec
model=Sequential()



embedding=Embedding(num_words,200,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(8))

model.add(LeakyReLU())

model.add(Dropout(rate = 0.1))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()
train1=tweet_pad[:train.shape[0]]

test1=tweet_pad[train.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train1,train['target'].values,test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
early = EarlyStopping(mode = 'min',monitor='X_test',patience=3)
history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=1,

                 callbacks=[early])
pre = model.predict(test1)
pre = np.round(pre).astype(int).reshape(3263)
sub=pd.DataFrame({'id':sub['id'].values.tolist(),'target':pre})

sub.to_csv('submission.csv',index=False)
from IPython.display import FileLink

FileLink(r'submission.csv')