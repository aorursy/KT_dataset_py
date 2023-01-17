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
EMBEDDING_DIM=256
WORD_MIN_COUNT=5
WINDOW=5
WORKER=4

UNK_TOKEN='<UNK>'
PAD_TOKEN='<PAD>'
# -*- coding: utf-8 -*-
#  @Time    : 2020-04-17 10:37
#  @Author  : Shupeng

import pandas as pd
import numpy as np
from gensim.models import Word2Vec, FastText


class WordEmbedding(object):
    def __init__(self, embedding_dim, window, min_count, worker):
        super(WordEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.window = window
        self.worker = worker
        self.word2vec = None
        self.fasttext = None
        self.glove = None

    def fit_word2vec(self, x):
        self.word2vec = Word2Vec(x,
                                 size=self.embedding_dim,
                                 window=self.window,
                                 min_count=self.min_count,
                                 workers=self.worker)

    def fit_fasttext(self, x):
        self.fasttext = FastText(x,
                                 size=self.embedding_dim,
                                 window=self.window,
                                 min_count=self.min_count,
                                 workers=self.worker)

    def save_word2vec(self, path):
        self.word2vec.save(path)

    def save_fasttext(self, path):
        self.fasttext.save(path)

    def load_word2vec(self, path):
        self.word2vec=Word2Vec.load(path)

    def load_fasttext(self, path):
        self.fasttext=FastText.load(path)
        
    def get_embedding_dim(self):
        return self.embedding_dim

    def get_word2vec_vectors(self):
        if self.word2vec is None:
            raise Exception("word2vec model is None")
        return self.word2vec.wv.vectors

    def get_word2vec_vocab(self):
        if self.word2vec is None:
            raise Exception("word2vec model is None")
        return self.word2vec.wv.vocab

    def get_word2vec_word2index(self):
        if self.word2vec is None:
            raise Exception("word2vec model is None")
        word2index = {}
        for i, w in enumerate(self.word2vec.wv.index2word):
            word2index[w] = i
        return word2index

    def get_word2vec_index2word(self):
        if self.word2vec is None:
            raise Exception("word2vec model is None")
        index2word = {}
        for i, w in enumerate(self.word2vec.wv.index2word):
            index2word[i] = w
        return index2word

    def get_fasttext_vectors(self):
        if self.fasttext is None:
            raise Exception("fasttext model is None")
        return self.fasttext.wv.vectors

    def get_fasttext_vocab(self):
        if self.fasttext is None:
            raise Exception("fasttext model is None")
        return self.fasttext.wv.vocab

    def get_fasttext_word2index(self):
        if self.fasttext is None:
            raise Exception("fasttext model is None")
        word2index = {}
        for i, w in enumerate(self.fasttext.wv.index2word):
            word2index[w] = i
        return word2index

    def get_fasttext_index2word(self):
        if self.fasttext is None:
            raise Exception("fasttext model is None")
        index2word = {}
        for i, w in enumerate(self.fasttext.wv.index2word):
            index2word[i] = w
        return index2word

    def word2vec_unk_and_pad(self, tokens, max_len, unk_token, pad_token):
        if self.word2vec is None:
            raise Exception("word2vec model is None")

        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        vocab = self.get_word2vec_vocab()
        tokens = [t if t in vocab else unk_token for t in tokens]
        tokens.extend([pad_token for i in range(len(tokens), max_len)])
        return tokens

    def retrain_word2vec(self, x):
        self.word2vec.build_vocab(x, update=True)
        self.word2vec.train(x, epochs=1, total_examples=self.word2vec.corpus_count)

    def fasttext_unk_and_pad(self, tokens, max_len, unk_token, pad_token):
        if self.fasttext is None:
            raise Exception("fasttext model is None")

        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        vocab = self.get_fasttext_vocab()
        tokens = [t if t in vocab else unk_token for t in tokens]
        tokens.extend([pad_token for i in range(len(tokens), max_len)])
        return tokens

    def retrain_fasttext(self, x):
        self.fasttext.build_vocab(x, update=True)
        self.fasttext.train(x, epochs=1, total_examples=self.fasttext.corpus_count)

# -*- coding: utf-8 -*-
#  @Time    : 2020-04-17 14:29
#  @Author  : Shupeng

import tensorflow as tf
VERBOSE = True

class TextCNN(tf.keras.models.Model):
    def __init__(self, embedding_matrix, max_length, filter_cnt, output_dim):
        super(TextCNN, self).__init__()
        self.vocab_size = len(embedding_matrix)
        self.embedding_dim = len(embedding_matrix[0])
        self.input_length = max_length
        self.filter_cnt = filter_cnt
        self.output_dim = output_dim
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size,
                                                         self.embedding_dim,
                                                         weights=[embedding_matrix],
                                                         trainable=True,
                                                         input_length=max_length)

        # 每一个卷积核从一句话里提取一个元素
        # 输入的shape为(batch_size, input_length, embedding_dim)
        # 输出的shape为(batch_size, input_length, filter_count)
        self.conv_1 = tf.keras.layers.Conv1D(filters=self.filter_cnt,
                                             kernel_size=2,
                                             padding='same',
                                             strides=1)

        self.conv_2 = tf.keras.layers.Conv1D(filters=self.filter_cnt,
                                             kernel_size=3,
                                             padding='same',
                                             strides=1)

        self.conv_3 = tf.keras.layers.Conv1D(filters=self.filter_cnt,
                                             kernel_size=4,
                                             padding='same',
                                             strides=1)

        # 因为都是一样的shape，都是(batch_size, input_length)
        # 而且max_pooling没有参数
        # 所以可以公用
        # 卷积层的每一个filter经过相同shape的max_pooling之后都只留下一个值
        # 相当于每个filter从一个sentence中提取一个最有价值的元素
        self.max_pooling = tf.keras.layers.MaxPool1D(self.input_length)
        # self.concat=tf.keras.layers.concatenate()
        self.flatten = tf.keras.layers.Flatten()
        # shape为(batch_size, filter_count)
        self.fc = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, x):
        emb_x = self.embedding_layer(x)

        conv_1_x = self.conv_1(emb_x)
        conv_2_x = self.conv_2(emb_x)
        conv_3_x = self.conv_3(emb_x)

        max_pool_1_x = self.max_pooling(conv_1_x)
        max_pool_2_x = self.max_pooling(conv_2_x)
        max_pool_3_x = self.max_pooling(conv_3_x)

        max_pool_x = tf.keras.layers.concatenate([max_pool_1_x, max_pool_2_x, max_pool_3_x])

        flatten_x = self.flatten(max_pool_x)

        logits = self.fc(flatten_x)

        if VERBOSE:
            print('emb_x', emb_x.shape)
            print('conv_1_x', conv_1_x.shape)
            print('max_pool_1_x', max_pool_1_x.shape)
            print('max_pool_x', max_pool_x.shape)
            print('flatten_x', flatten_x.shape)
            print('logits', logits.shape)

        return logits

df=pd.read_csv('/kaggle/input/cutted_df.csv')
df.dropna(how='any',inplace=True)
df['quiz']=df['quiz'].apply(lambda x:x.split())
df.head()
word_emb = WordEmbedding(EMBEDDING_DIM, WINDOW, WORD_MIN_COUNT, WORKER)
# first train of word embedding
word_emb.fit_fasttext(df['quiz'].values)
word_emb.fit_word2vec(df['quiz'].values)
# get max length
len_dis = df['quiz'].apply(lambda x: len(x))
max_len = int(np.ceil(np.mean(len_dis) + 3 * np.std(len_dis)))
# pas sequence and convert words not in vocab into unknown token
df['ft_quiz'] = df['quiz'].apply(lambda x: word_emb.fasttext_unk_and_pad(x, max_len, UNK_TOKEN, PAD_TOKEN))
df['wv_quiz'] = df['quiz'].apply(lambda x: word_emb.word2vec_unk_and_pad(x, max_len, UNK_TOKEN, PAD_TOKEN))
# train the second time to include <PAD> and <UNK>
word_emb.retrain_fasttext(df['ft_quiz'].values)
word_emb.retrain_word2vec(df['wv_quiz'].values)
ft_word2index = word_emb.get_fasttext_word2index()
wv_word2index = word_emb.get_word2vec_word2index()
# convert word into index
df['ft_quiz'] = df['ft_quiz'].apply(lambda x: [ft_word2index[t] for t in x])
df['wv_quiz'] = df['wv_quiz'].apply(lambda x: [wv_word2index[t] for t in x])
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split

lb=LabelBinarizer()
y_subject=lb.fit_transform(df['subject']).tolist()
subject_labels=lb.classes_
y_course=lb.fit_transform(df['course']).tolist()
course_labels=lb.classes_

X,y=df['ft_quiz'].values.tolist(),y_subject
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_train,X_dev,y_train,y_dev=train_test_split(X_train,y_train,test_size=0.1,random_state=42,stratify=y_train)
y_subject
text_cnn=TextCNN(word_emb.get_fasttext_vectors(),max_len,100,len(subject_labels))
text_cnn.compile(tf.optimizers.Adam(learning_rate=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
print('train')
early_stopping=tf.keras.callbacks.EarlyStopping(patience=10,mode='max')

history=text_cnn.fit(X_train,y_train,
                    batch_size=512,
                    epochs=200,
                    callbacks=[early_stopping],
                    validation_data=(X_dev,y_dev))
pred_probas=text_cnn.predict(X_test)
y_pred=np.argmax(pred_probas,axis=1)
y_true=np.argmax(y_test,axis=1)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_true,y_pred))
conf_matrix=confusion_matrix(y_true,y_pred)
print(conf_matrix)
X,y=df['ft_quiz'].values.tolist(),y_course
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_train,X_dev,y_train,y_dev=train_test_split(X_train,y_train,test_size=0.1,random_state=42,stratify=y_train)

text_cnn=TextCNN(word_emb.get_fasttext_vectors(),max_len,100,len(course_labels))
text_cnn.compile(tf.optimizers.Adam(learning_rate=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print('train')
early_stopping=tf.keras.callbacks.EarlyStopping(patience=10,mode='max')

history=text_cnn.fit(X_train,y_train,
                    batch_size=512,
                    epochs=200,
                    callbacks=[early_stopping],
                    validation_data=(X_dev,y_dev))

pred_probas=text_cnn.predict(X_test)
y_pred=np.argmax(pred_probas,axis=1)
y_true=np.argmax(y_test,axis=1)
print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
