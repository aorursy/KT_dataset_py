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
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
%matplotlib inline
#nlp imports
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
#Models and Layers imports
import tensorflow as tf
from keras.layers import Dense,Embedding,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
testdf = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print("Train Data: ",df.shape)
print("Test Data: ",testdf.shape)
df.head()
# Count of Disaster/ Non Disaster tweet
tmp = df.groupby('target').count()['text']
tmp
# target Distribution by Keywords
tmp = pd.DataFrame()
tmp['target_mean'] = df.groupby('keyword')['target'].transform('mean')
tmp['keyword'] = df['keyword']
tmp['target'] = df['target']

fig = plt.figure(figsize=(8, 78), dpi=100)

sns.countplot(y=tmp.sort_values(by='target_mean', ascending=False)['keyword'],
              hue=tmp.sort_values(by='target_mean', ascending=False)['target'])

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=1)
plt.title('Target Distribution in Keywords')

plt.show()

#Max Length of the tweet in both target
tmp = pd.DataFrame()
tmp['len'] = df['text'].str.split().map(lambda x : len(x))
tmp['target'] = df['target']
print("For 1: ", max(tmp[tmp['target']==1]['len']))
print("For 0: ", max(tmp[tmp['target']==0]['len']))
def fillna(df):
    #reuse them for any new data
    for col in ['keyword','location']:
        df[col] = df[col].fillna(f'no_{col}')
fillna(df)
df.head()
def removetags(df):
    df['text'] = df['text'].str.replace('https?://\S+|www\.\S+','').str.replace('<.*?>','')
def preprocesstweet(text):
    emoji = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji.sub(r'',text)
    
    table = str.maketrans('','',string.punctuation)
    text = text.translate(table)
    
    text = re.sub('\s+', ' ', text).strip() 
    text = text.lower()
    new = ''
    stop_words = set(stopwords.words("english"))
    for w in word_tokenize(text):
        if w not in stop_words:
            new += w
            new += ' '
    return new
def applypre(df):
    removetags(df)
    df['text'] = df['text'].apply(lambda x: preprocesstweet(x))
applypre(df)
applypre(testdf)
#Dimension of the embedding vector
end_dim = 100
#Getting the Data Embedding matrix
lines = list(df['text'].values)
tweets = list()
for line in lines:
    tweets.append(word_tokenize(line))
print(tweets[:3])
model = Word2Vec(sentences=tweets,
                size = end_dim,
                window = 5,
                workers = 4,
                min_count = 1)
words = list(model.wv.vocab)
print(len(words))
words[:10]
fname = "/kaggle/working/wv.txt"
model.wv.save_word2vec_format(fname,binary=False)
!cat wv.txt | head -2
emd_idx = {}
f = open("/kaggle/working/wv.txt",encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coef = values[1:]
    emd_idx[word] = coef
f.close()
tkn = Tokenizer()
tkn.fit_on_texts(tweets)
seq = tkn.texts_to_sequences(tweets)

word_idx = tkn.word_index #index of words table  for the tweets
tweet_pad = pad_sequences(seq,maxlen=31) #we saw this above
def createrunnableinput(df):
    #for final test data predictions
    lines = list(df['text'].values)
    tweets = list()
    for line in lines:
        tweets.append(word_tokenize(line))
    tkn = Tokenizer()
    tkn.fit_on_texts(tweets)
    seq = tkn.texts_to_sequences(tweets)

    word_idx = tkn.word_index #index of words table  for the tweets
    tweet_pad = pad_sequences(seq,maxlen=31) #we saw this above
    
    return tweet_pad
target = df['target'].values
num_words = len(word_idx)+1
emd_matrix = np.zeros((num_words,end_dim))
for word, i in word_idx.items():
    if i>num_words:
        continue
    emd_vector = emd_idx[word]
    if emd_vector is not None:
        emd_matrix[i] = emd_vector
model = Sequential()
emd_layer = Embedding(num_words,
                     end_dim,
                     embeddings_initializer=Constant(emd_matrix),
                     input_length = 31,
                     trainable = False)
model.add(emd_layer)
model.add(GRU(units = 32,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss = "binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()
train_split = 0.8
indices = np.arange(tweet_pad.shape[0])
np.random.shuffle(indices)

tweet_pad = tweet_pad[indices]
target = target[indices]

train_samples = int(tweet_pad.shape[0]*train_split)

Xtrain = tweet_pad[:train_samples]
ytrain = target[:train_samples]

Xval = tweet_pad[train_samples:]
yval = target[train_samples:]
model.fit(Xtrain,ytrain,batch_size=32,epochs=20,validation_data=(Xval,yval))
xtest = createrunnableinput(testdf)
ypred = model.predict(xtest)
submit = pd.DataFrame()
submit['id'] = testdf['id']
submit['target'] = ypred
submit['target'] = submit['target'].apply(lambda x: 0 if x<0.5 else 1)
submit.head()
submit.to_csv('submission.csv',index=False)
