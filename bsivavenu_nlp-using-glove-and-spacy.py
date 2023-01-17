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
import pandas as pd,numpy as np,seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline
import spacy
yelp_reviews = pd.read_csv('../input/yelp-dataset/yelp_review.csv',nrows=10000)
yelp_reviews.head()
yelp_reviews.columns
yelp_reviews=yelp_reviews.drop(['review_id','user_id','business_id','date','useful','funny','cool'],axis=1)
yelp_reviews.head()
yelp_reviews.isnull().any()
yelp_reviews.stars.unique()
sns.countplot(yelp_reviews.stars)
yelp_reviews.stars.mode()
reviews = yelp_reviews[yelp_reviews.stars!=3]

reviews['label'] = reviews['stars'].apply(lambda x: 1 if x>3 else 0)
reviews = reviews.drop('stars',axis=1)
reviews.head()
reviews.shape
text = reviews.text.values
label = reviews.label.values
nlp = spacy.load('en')
text[0]
parsed_text = nlp(text[0])
parsed_text
for i,sentance in enumerate(parsed_text.sents):
    print(i,':',sentance)
for num, entity in enumerate(nlp(text[10]).ents):
    print ('Entity {}:'.format(num + 1), entity, '-', entity.label_)
token_pos = [token.pos_ for token in nlp(text[10])]
tokens = [token for token in nlp(text[10])]
sd = list(zip(tokens,token_pos))
sd = pd.DataFrame(sd,columns=['token','pos'])
sd.head()
max_num_words = 1000
max_seq_length = 100
tokenizer = Tokenizer(num_words=max_num_words)

len(yelp_reviews)
reviews=yelp_reviews[:100000]
reviews=reviews[reviews.stars!=3]

reviews["labels"]= reviews["stars"].apply(lambda x: 1 if x > 3  else 0)
reviews=reviews.drop("stars",axis=1)

reviews.head()
texts = reviews["text"].values
labels = reviews["labels"].values
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

len(word_index)
data = pad_sequences(sequences, maxlen=max_seq_length)
data
data.shape
labels = to_categorical(np.asarray(labels))
labels.shape
validation_spilit = 0.2
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
data
labels = labels[indices]
labels
nb_validation_samples = int(validation_spilit*data.shape[0])
nb_validation_samples
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

glove_dir = '../input/glove-global-vectors-for-word-representation/'
embedding_index = {}

f = open(os.path.join(glove_dir,'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embedding_index[word] = coefs
f.close()

print('found word vecs: ',len(embedding_index))
embedding_dim = 50
embedding_matrix = np.zeros((len(word_index)+1,embedding_dim))
embedding_matrix.shape
for word,i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=max_seq_length,trainable=False)
from keras.layers import Bidirectional,GlobalMaxPool1D,Conv1D
from keras.layers import LSTM,Input,Dense,Dropout,Activation
from keras.models import Model
inp = Input(shape=(max_seq_length,))
x = embedding_layer(inp)
x = Bidirectional(LSTM(50,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50,activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(2,activation='sigmoid')(x)
model = Model(inputs=inp,outputs=x)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=20,batch_size=128);
score = model.evaluate(x_val,y_val)
score
score[1]*100