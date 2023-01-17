import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy import array

from numpy import asarray

from numpy import zeros

import matplotlib.pyplot as plt

import seaborn as sns

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

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')

tweet.head(3)
print('There are {} rows and {} columns in train'.format(tweet.shape[0],tweet.shape[1]))

print('There are {} rows and {} columns in train'.format(test.shape[0],test.shape[1]))
x=tweet.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=tweet[tweet['target']==1]['text'].str.len()

ax1.hist(tweet_len,color='red')

ax1.set_title('disaster tweets')

tweet_len=tweet[tweet['target']==0]['text'].str.len()

ax2.hist(tweet_len,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=tweet[tweet['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='red')

ax1.set_title('disaster tweets')

tweet_len=tweet[tweet['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Words in a tweet')

plt.show()
df=pd.concat([tweet,test])

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



remove_emoji("Omg another Earthquake 😔😔")
df['text']=df['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))
unique_words = df['text'].str.split(expand=True).stack().value_counts() 

print(len(unique_words))
def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus
corpus=create_corpus(df)
embedding_dict={}

with open('../input/glove6b100dtxt/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN = 140

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_dim = 100

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec
train=tweet_pad[:tweet.shape[0]]

test=tweet_pad[tweet.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
from keras.models import Sequential

from keras import layers

from keras.regularizers import l2

from tensorflow.python.keras.losses import BinaryCrossentropy



model = Sequential()

model.add(layers.Embedding(num_words, embedding_dim, 

                           weights=[embedding_matrix], 

                           input_length=MAX_LEN, 

                           trainable=True))

model.add(layers.Flatten())

model.add(layers.Dropout(0.3))

model.add(layers.Dense(500, activation='relu'))

model.add(layers.Dense(250, activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(100, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=10,

                    verbose=1,

                    validation_data=(X_test,y_test),

                    batch_size=32)
loss, accuracy = model.evaluate(X_train, y_train, verbose=True)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=True)

print("Testing Accuracy:  {:.4f}".format(accuracy))
predicted = model.predict(X_test)

predicted_train = model.predict(X_train)
predicted_class = np.round(predicted)

predicted_train = np.round(predicted_train)
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy:", accuracy_score(y_test, predicted_class))

print("f1_score", f1_score(y_test, predicted_class, average = 'micro'))
import sys

import numpy as np

import keras

import pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D,LSTM, AveragePooling2D

from keras.layers import MaxPool1D

from keras.models import Model

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam
# filter sizes of the different conv layers 

filter_sizes = [1,2,3]

num_filters = 128

embedding_dim = 100

# dropout probability

drop = 0.2

batch_size = 30

epochs = 10
############# Keras Embedding #######

embedding_layer = Embedding(num_words,

                            embedding_dim,

                            weights=[embedding_matrix],

                            input_length=MAX_LEN,

                            trainable=True)
#################### Model Structure ###############

inputs = Input(shape=(MAX_LEN,), dtype='int32')

embedding = embedding_layer(inputs)



print(embedding.shape)

reshape = Reshape((MAX_LEN,embedding_dim,1))(embedding)

print(reshape.shape)



conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

#conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

#conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[4], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)



maxpool_0 = MaxPool2D(pool_size=(MAX_LEN - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)

maxpool_1 = MaxPool2D(pool_size=(MAX_LEN - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)

maxpool_2 = MaxPool2D(pool_size=(MAX_LEN - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

#maxpool_3 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[3] + 1, 1), strides=(1,1), padding='valid')(conv_3)

#maxpool_4 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[4] + 1, 1), strides=(1,1), padding='valid')(conv_4)



concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

flatten = Flatten()(concatenated_tensor)

dropout = Dropout(drop)(flatten)

layer_1 = Dense(units=500, activation='relu')(flatten)

#layer_2 = Dense(units=500, activation='relu')(layer_1)

output = Dense(units=1, activation='sigmoid')(layer_1)
# this creates a model that includes

model = Model(inputs=inputs, outputs=output)



checkpoint = ModelCheckpoint('weights_cnn_sentece.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(X_train

                       , y_train

                       , epochs=4

                       , batch_size=32

                       , validation_data = (X_test,y_test)

                       , verbose=1)
predicted = model.predict(X_test)

predicted_train = model.predict(X_train)
predicted_class = np.round(predicted)

predicted_train = np.round(predicted_train)
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy:", accuracy_score(y_test, predicted_class))

print("f1_score", f1_score(y_test, predicted_class, average = 'micro'))
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
history=model.fit(X_train,y_train,

                  batch_size=4,

                  epochs=15,

                  validation_data=(X_test,y_test),verbose=2)
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
y_pre=model.predict(test)

y_pre=np.round(y_pre).astype(int).reshape(3263)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})

sub.to_csv('submission.csv',index=False)
sub.head()