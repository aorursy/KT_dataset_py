import os
import numpy as np
import pandas as pd
import nltk
import keras
import six
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,AveragePooling1D,Flatten,concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import *
import h5py
train_df = pd.read_csv('../input/noticias-falsas-en-espaol/train.csv')
train_df.head()
test_df = pd.read_csv('../input/noticias-falsas-en-espaol/test.csv')
test_df.head()
articles = train_df['text'].values.tolist()

titles = train_df['title'].values.tolist()

labels = train_df['label'].values.tolist()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('spanish'))
article_w = []
labels_up = []
tokenizer = RegexpTokenizer(r'\w+')

for i,each_article in enumerate(articles):
    if(isinstance(each_article, six.string_types)):
        words = tokenizer.tokenize(each_article)
        article_w.append(words)
        labels_up.append(labels[i])
for i,each_article in enumerate(article_w):
    for word in each_article:
        if word in stop_words:
            each_article.remove(word)
    article_w[i] = each_article
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
lemmatizer = WordNetLemmatizer()
for i,each_article in enumerate(article_w):
    for j,word in enumerate(each_article):
        each_article[j] = lemmatizer.lemmatize(word, get_wordnet_pos(word))
    article_w[i] = each_article
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
glove = gensim.models.KeyedVectors.load_word2vec_format('../input/pretrained-word-vectors-for-spanish/SBW-vectors-300-min5.txt')
word2index = {token: token_index for token_index, token in enumerate(glove.index2word)}
wordEmbeddings = [glove[key] for key in word2index.keys()]
word2index["PADDING"] = len(word2index)
wordEmbeddings.append(np.zeros(len(wordEmbeddings[0])))
word2index["UNKNOWN"] = len(word2index)
wordEmbeddings.append(np.random.uniform(-0.25, 0.25, len(wordEmbeddings[0])))
wordEmbeddings = np.array(wordEmbeddings)
wordEmbeddings.shape
def embed(articles):
    embedded = []
    for article in articles:
        a_idx = []
        for word in article:
            if word in word2index.keys():
                a_idx.append(word2index[word])
            elif word.lower() in word2index.keys():
                a_idx.append(word2index[word.lower()])
            else:
                a_idx.append(word2index['UNKNOWN'])
        embedded.append(a_idx)
    return embedded
def find_max_len(articles):
    maxlen = 0
    tot = 0
    for article in articles:
        if len(article)>maxlen:
            maxlen = len(article)
        tot = tot + len(article)
    avg = tot/len(articles)
    return avg,maxlen
def padding(articles, maxm):
    padded = pad_sequences(articles, 500, padding='post', value=word2index['PADDING'])
    return padded
avg, maxm = find_max_len(article_w)
articles = embed(article_w)
articles = padding(articles, maxm)
avg
x_train, x_test, y_train, y_test = train_test_split(articles, labels_up, test_size=0.2)
x_train.shape
words_input = Input(shape=(500,),dtype='int32',name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings], trainable=False)(words_input)
conv_1 = Conv1D(filters=100, kernel_size=10, strides=2, activation='relu')(words)
avgpool_1 = AveragePooling1D(pool_size=10, strides=10)(conv_1)
b_lstm = Bidirectional(LSTM(200, activation='tanh', return_sequences=False))(avgpool_1)
dense_1 = Dense(128, activation='relu')(b_lstm)
dropout = Dropout(0.1)(dense_1)
dense_2 = Dense(1, activation='relu')(dropout)


sgd = keras.optimizers.adam(lr=0.0001)
model = Model(inputs=words_input, outputs=dense_2)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
model.summary()
history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=10, validation_split=0.20)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='cv')
plt.title('Loss count')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='cv')
plt.title('Loss count')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
pred = model.predict(x_test)
for i,p in enumerate(pred):
    if p>0.5:
        pred[i] = 1
    else:
        pred[i] = 0
from sklearn.metrics import f1_score
f1 = f1_score(y_test, pred)
print(f1)
