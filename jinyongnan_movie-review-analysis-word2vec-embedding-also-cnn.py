#https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456

EMBED_DIM = 100
import pandas as pd

import numpy as np

df = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv",encoding = "utf-8")

df["sentiment"] = (df["sentiment"]=="positive").astype("int8")

x_train = df["review"].iloc[:25000].values

y_train = df["sentiment"].iloc[:25000].values

x_test = df["review"].iloc[25000:].values

y_test = df["sentiment"].iloc[25000:].values
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()

total_reviews = x_train + x_test

tokenizer.fit_on_texts(total_reviews)

max_length = max([len(review.split()) for review in total_reviews])

x_train_tokens = tokenizer.texts_to_sequences(x_train)

x_test_tokens = tokenizer.texts_to_sequences(x_test)

vocab_size = len(tokenizer.word_index) + 1

x_train_pad = pad_sequences(x_train_tokens,maxlen=max_length,padding="post")

x_test_pad = pad_sequences(x_test_tokens,maxlen=max_length,padding="post")

print("max_len:%d,vocab_size:%d"%(max_length,vocab_size))
# print("test:",x_test[0])

# print("train:",x_train[0])

# print("+:",(x_test+x_train)[0])
import string

print("punctuation:",string.punctuation)

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

print("stopwords:",set(stopwords.words("english")))

lines = df["review"].values.tolist()

stop_words = set(stopwords.words("english"))

reviews = list()

for line in lines:

    tokens = word_tokenize(line)

    tokens = [w.lower() for w in tokens]

    table = str.maketrans("","",string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    words = [w for w in stripped if w.isalpha()]

    words = [w for w in words if w not in stop_words]

    reviews.append(words)

len(reviews)
import gensim

model = gensim.models.Word2Vec(sentences=reviews,size=EMBED_DIM,window=5,workers=4,min_count=1)
words=list(model.wv.vocab)

print("vocabulary size:",len(words))

model.wv.get_vector(words[1])
tokenizer = Tokenizer()

tokenizer.fit_on_texts(reviews)

seqs = tokenizer.texts_to_sequences(reviews)

review_pad = pad_sequences(seqs,maxlen=max_length,padding="post")

word_index = tokenizer.word_index

sentiments = df["sentiment"].values



num_words = len(word_index)+1

embedding_matrix = np.zeros((num_words,EMBED_DIM))

for word,i in word_index.items():

    vector = model.wv.get_vector(word)

    if vector is not None:

        embedding_matrix[i] = vector
embedding_matrix
from keras.models import Sequential

from keras.layers import Dense,Embedding,LSTM,GRU,Flatten,Dropout

from keras.layers.embeddings import Embedding

from keras.initializers import Constant

from keras.layers.convolutional import Conv1D,MaxPooling1D



model_main = Sequential()

model_main.add(Embedding(num_words,EMBED_DIM,input_length=max_length))#,embeddings_initializer=Constant(embedding_matrix),trainable=False))

model_main.add(LSTM(32,dropout=0.2,recurrent_dropout=0.2))

model_main.add(Dense(32,activation="relu"))

model_main.add(Dense(1,activation="sigmoid"))

model_main.compile(loss="binary_crossentropy",optimizer = "adam",metrics=["acc"])
model_main.summary()
model_main.fit(review_pad,sentiments,batch_size=128,epochs=10,validation_split=0.25)