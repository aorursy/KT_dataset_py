import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam



import os

#os.listdir('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')
df_train= pd.read_csv('../input/nlp-getting-started/train.csv')

df_test=pd.read_csv('../input/nlp-getting-started/test.csv')

df_train.head(3)
print('There are {} rows and {} columns in train'.format(df_train.shape[0],df_train.shape[1]))

print('There are {} rows and {} columns in test'.format(df_test.shape[0],df_test.shape[1]))
x=df_train.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_length=df_train[df_train['target']==1]['text'].str.len()

ax1.hist(tweet_length,color='red')

ax1.set_title('disaster tweets')

tweet_length=df_train[df_train['target']==0]['text'].str.len()

ax2.hist(tweet_length,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Number of Characters in Tweets')

plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_length=df_train[df_train['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_length,color='red')

ax1.set_title('disaster tweets')

tweet_length=df_train[df_train['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_length,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Number of Words in a tweet')

plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

word_length=df_train[df_train['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word_length.map(lambda x: np.mean(x)),ax=ax1,color='red')

ax1.set_title('disaster')

word_length=df_train[df_train['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word_length.map(lambda x: np.mean(x)),ax=ax2,color='green')

ax2.set_title('Not disaster')

fig.suptitle('Average word length in each tweet')
def create_corpus(target):

    corpus=[]

    

    for x in df_train[df_train['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus

# this function collects all of the words in all of the tweets with target = 1 (disaster) or 0 (no disaster)

# print(create_corpus(1)[5:1300])



# important note: removing hashtag symbol might not be a good idea. Because #earthquake might strongly indicate that 

# there was an earthquake and people are tweeting about it.
corpus=create_corpus(0)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

print(top)

# I don't know what stopwords are. Are they irrelevant words?
x,y=zip(*top)

plt.bar(x,y)
corpus=create_corpus(1)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1



top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

    





x,y=zip(*top)

plt.bar(x,y)
plt.figure(figsize=(10,5))

corpus=create_corpus(1)



dic=defaultdict(int)

import string

special = string.punctuation

for word in (corpus):

    for char in word:

        if char in special:

            dic[char]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y)

print(dic["#"])

# hashtags can be alerts for disasters. be careful!

#### WRONG! This is looking for separate hashtag signs. something like #earthquake not counted.
plt.figure(figsize=(10,5))

corpus=create_corpus(0)



dic=defaultdict(int)

import string

special = string.punctuation

for word in (corpus):

    for char in word:

        if char in special:

            dic[char]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y,color='green')

print(dic["#"])



counter=Counter(corpus)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:40]:

    if (word not in stop) :

        x.append(word)

        y.append(count)
sns.barplot(x=y,y=x)
def get_top_tweet_bigrams(bigram_corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(bigram_corpus)

    bag_of_words = vec.transform(bigram_corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10,5))

top_tweet_bigrams=get_top_tweet_bigrams(df_train['text'])[:10]

x,y=map(list,zip(*top_tweet_bigrams))

sns.barplot(x=y,y=x)

print(get_top_tweet_bigrams(df_train['text'])[:40])

# Most of these bigrams are meaningless I think. A few are important like: 'burning buildings' or 'suicide bomber'
df=pd.concat([df_train,df_test])

df.shape
example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



remove_URL(example)
df['text']=df['text'].apply(lambda x : remove_URL(x))
example = """<div>

<h1>Real or Fake</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

print(remove_html(example))
df['text']=df['text'].apply(lambda x : remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

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



remove_emoji("Omg another Earthquake ????????")
df['text']=df['text'].apply(lambda x: remove_emoji(x))

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))
!pip install pyspellchecker
from spellchecker import SpellChecker



spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)

        

text = "corect me plese"

correct_spellings(text)
#df['text']=df['text'].apply(lambda x : correct_spellings(x)#)

# why is this commented?


def create_corpus(df):

    corpus=[]

    for string in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(string) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus

# tqdm is for progress bar        

        
corpus=create_corpus(df)
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

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

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec

            
model=Sequential()



embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])



model.summary()
train=tweet_pad[:df_train.shape[0]]

test=tweet_pad[df_train.shape[0]:]

# local testing of the model
X_train,X_test,y_train,y_test=train_test_split(train,df_train['target'].values,test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=4,epochs=3,validation_data=(X_test,y_test),verbose=2)

# increase the epochs to 15 or more once you are getting satisfactory results
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
y_pre=model.predict(test)

y_pre=np.round(y_pre).astype(int).reshape(3263)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})

sub.to_csv('submission.csv',index=False)

sub.head(60)