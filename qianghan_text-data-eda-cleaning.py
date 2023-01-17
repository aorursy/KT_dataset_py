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
!pip install pyspellchecker
from scipy import *

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords

from collections import defaultdict

import string

from collections import Counter

from spellchecker import SpellChecker

from tqdm import tqdm  #Instantly make your loops show a smart progress meter - just wrap any iterable with tqdm(iterable)

from nltk.tokenize import word_tokenize

import re







train= pd.read_csv('../input/nlp-getting-started/train.csv')

print(train.shape)

train.head()

test=pd.read_csv('../input/nlp-getting-started/test.csv')

print(test.shape)

test.head()

x=train.target.value_counts()

sns.barplot(x.index,x)

plt.show()
pd.isnull(train).sum()
pd.isnull(test).sum()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

train_len=(train.text[train.target==1]).str.len()

ax1.hist(train_len,color='red')

ax1.set_title('disaster tweets')

train_len=(train.text[train.target==0]).str.len()

ax2.hist(train_len,color='green')

ax2.set_title('not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

train_len=(train.text[train.target==1]).str.split().map(lambda x: len(x))

ax1.hist(train_len,color='red')

ax1.set_title('disaster tweets')

train_len=(train.text[train.target==0]).str.split().map(lambda x: len(x))

ax2.hist(train_len,color='green')

ax2.set_title('not disaster tweets')

fig.suptitle('Words in tweets')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

word=(train.text[train.target==1]).str.split().map(lambda x : [len(i) for i in x])

ave_wordlen=word.map(lambda x:mean(x))

sns.distplot(ave_wordlen,ax=ax1,color='red')

ax1.set_title('disaster tweets')



#print(ave_wordlen)

word=(train.text[train.target==0]).str.split().map(lambda x : [len(i) for i in x])

ave_wordlen=word.map(lambda x:mean(x))

sns.distplot(ave_wordlen,ax=ax2,color='green')

ax2.set_title('not disaster tweets')

fig.suptitle('Average word length  in each tweet')

plt.show()
def create_corpus(target):

    corpus=[]

    for item in (train.text[train.target==target]).str.split():

        for x in item:

            corpus.append(x)

    return corpus

corpus=create_corpus(0)

print(corpus)

print('length of the corpus:',len(corpus))



s='mississippi'

d=defaultdict(int)

for k in s:

    d[k]+=1

print(d,d.keys(),d.items(),d['i'])

print('num of stop words:',len(stopwords.words('english')))

print('num of stop words:',len(set(stopwords.words('english'))))
stop=stopwords.words('english')

dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

#print(dic.items())

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

#print(top)

    
print(top,type(top))

print(*top)

print(zip(*top))

x,y=zip(*top)

print(x,y)

plt.bar(x,y)



corpus=create_corpus(1)

special=string.punctuation

print(special,type(special))

dic=defaultdict(int)

for word in corpus:

    if word in special:

        dic[word]+=1

print(*dic.items())

x,y=zip(*dic.items())

plt.bar(x,y)

plt.show()
fig=plt.subplots(1,1,figsize=(10,5))



count=Counter(corpus)

most=count.most_common(50)

x,y=zip(*most)

commons,counts=[],[]

for idx,item in enumerate(x):

    if item not in stop:

        commons.append(item)

        counts.append(y[idx])

        

plt.bar(commons,counts)

plt.show()
df=pd.concat([train,test])

df.shape

example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

   # url = re.compile(r'https?://\S+')  ##\S means matching any string and \S+ is just a repetion of \S.



    return url.sub(r'',text)



remove_URL(example)
df.text=(df.text).apply(lambda x:remove_URL(x))

df.head()

example = """<div>

<h1>Real or Fake</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""

def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

print(remove_html(example))
df.text=(df.text).apply(lambda x:remove_html(x))

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
df['text']=df['text'].apply(lambda x: remove_emoji(x))

def remove_punct(text):

    table=str.maketrans('','',string.punctuation) #maketrans(str1, str2, str3) 

    #str1 : Specifies the list of characters that need to be replaced.

    #str2 : Specifies the list of characters with which the characters need to be replaced.

    #str3 : Specifies the list of characters that needs to be deleted.

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))
table=str.maketrans('','',string.punctuation)

df['text']=df['text'].apply(lambda x: remove_punct(x))



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
#df['text']=df['text'].apply(lambda x: correct_spellings(x))

def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)  ##str.isalpha() True- If all characters in the string are alphabet False otherwise

    return corpus
corpus=create_corpus(df)



MAX_LEN=50

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')

print(corpus[0],corpus[1])

print(sequences[0],sequences[1])

print(tweet_pad.shape,len(corpus))

print(tweet_pad)
word_index=tokenizer_obj.word_index

print(word_index)

print('Number of unique words:',len(word_index))
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
print(list(embedding_dict.keys())[:10])

print(list(embedding_dict.items())[:10])
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100)) ##100 because we used Glove of 100d



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec
from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

model=Sequential()



embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])

print(model.summary)



train_emb=tweet_pad[:train.shape[0]]

test_emb=tweet_pad[train.shape[0]:]

#print(train.shape,train)

X_train,X_test,y_train,y_test=train_test_split(train_emb,train['target'],test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)

history=model.fit(X_train,y_train,batch_size=4,epochs=2,validation_data=(X_test,y_test),verbose=2)

y_pre=model.predict(test_emb)

y_pre=np.round(y_pre).astype(int)