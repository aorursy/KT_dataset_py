# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
train.head()
train.info()
# missing values
train.isnull().sum()
test.isnull().sum()
# check the percentage of missing values

# train
test_miss_percent = train.isnull().mean()*100
print(test_miss_percent)
print()

# test
train_miss_percent = test.isnull().mean()*100
print(train_miss_percent)
# none-disastrous
train[train.target == 0]['text'][:5]
# disastrous
train[train.target == 1]['text'][:5]
train['target'].value_counts()
sns.barplot(train['target'].value_counts().index, train['target'].value_counts())
train['keyword'].nunique()
figure = plt.figure(figsize=(14, 10))
sns.barplot(y=train['keyword'].value_counts().index[:20], x=train['keyword'].value_counts()[:20])
train["location"].nunique()
figure = plt.figure(figsize=(14, 10))
sns.barplot(y=train['location'].value_counts().index[:20], x=train['location'].value_counts()[:20])
figure = plt.figure(figsize=(14, 10))
sns.barplot(y=train['location'].value_counts().index[-20:], x=train['location'].value_counts()[-20:])
import re
import string
# lowercase
def lowercase_text(text):
    return text.lower()


# remove
def remove_noise(text):
    text = re.sub('\[.*?]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    """
    String of ASCII characters which are considered punctuation characters 
    in the C locale: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.
    """
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)    
    text = re.sub('\n', '', text)    
    text = re.sub('\w*\d\w*', '', text)
    return text
train['text'] = train['text'].apply(lambda x: lowercase_text(x))
test['text'] = test['text'].apply(lambda x: lowercase_text(x))
train['text'].head()
train['text'] = train['text'].apply(lambda x: remove_noise(x))
test['text'] = test['text'].apply(lambda x: remove_noise(x))
train['text'].head()
! pip install nlppreprocess
from nlppreprocess import NLP

nlp = NLP()

train['text'] = train['text'].apply(nlp.process)
test['text'] = test['text'].apply(nlp.process)
train['text'].head()
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(text)

train['text'] = train['text'].apply(stemming)
test['text'] = test['text'].apply(stemming)
train['text'][0]
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Using CountVectorizer to change the teweets to vectors
count_vectorizer_one_hot = CountVectorizer(analyzer='word', binary=True)
count_vectorizer_one_hot.fit(train['text'])

train_vectors = count_vectorizer_one_hot.fit_transform(train['text'])
test_vectors = count_vectorizer_one_hot.transform(test['text'])


# Printing first vector
print(train_vectors[0].toarray())

y = train['target']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_text = train_test_split(train_vectors, y, test_size=0.3, random_state=22)

clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = accuracy_score(y_pred, y_text)
print(score)
count_vectorizer = CountVectorizer(binary=False)
count_vectorizer.fit(train['text'])

train_vectors = count_vectorizer.fit_transform(train['text'])
test_vectors = count_vectorizer.transform(test['text'])


# Printing first vector
print(train_vectors[0].toarray())

y = train['target']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_text = train_test_split(train_vectors, y, test_size=0.3, random_state=22)

clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = accuracy_score(y_pred, y_text)
print(score)
tf_idf_vectorizer = TfidfVectorizer()
tf_idf_vectorizer.fit(train['text'])

train_vectors = tf_idf_vectorizer.fit_transform(train['text'])
test_vectors = tf_idf_vectorizer.transform(test['text'])


# Printing first vector
print(train_vectors[0].toarray())

y = train['target']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_vectors, y, test_size=0.3, random_state=22)

clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = accuracy_score(y_pred, y_test)
print(score)
bi_gram_vectorizer = CountVectorizer(ngram_range=(2,2), binary=True)
bi_gram_vectorizer.fit(train['text'])

train_vector = bi_gram_vectorizer.fit_transform(train['text'])
test_vector = bi_gram_vectorizer.fit_transform(test['text'])

clf = LogisticRegression(penalty='l2')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(y_pred)
score = accuracy_score(y_pred, y_text)
print(score)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow import keras

def build_model(
    vocab_size: int,
    output_size: int = 2,
    hidden_size: int = 16,
    optimizer: str = "adam",
    metrics: str = "accuracy"
):
    """
    use functional api model
    """
    x = Input(vocab_size)
    dense = Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01), name="dense1")(x)
    dense = Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name="dense2")(x)
    output = Dense(output_size, name='output')(x)
    model = Model(inputs=[x], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[metrics])
    
    return model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# vectrize dataset
tf_idf_vectorizer = TfidfVectorizer()
tf_idf_vectorizer.fit(train['text'])

train_vectors = tf_idf_vectorizer.fit_transform(train['text'])
test_vectors = tf_idf_vectorizer.transform(test['text'])

train_array = train_vectors.toarray()
test_array = test_vectors.toarray()

x_train, x_test, y_train, y_test = train_test_split(train_array, y, test_size=0.3, random_state=22)


# build model
vocab_size = len(tf_idf_vectorizer.vocabulary_)
label_size = len(set(y))

model = build_model(vocab_size=vocab_size, output_size=label_size)

# callback function
callbacks = [
    EarlyStopping(patience=5),
    ModelCheckpoint('model.h5', save_best_only=True),
    TensorBoard(log_dir='logs')
]

history = model.fit(x_train, y_train,
         validation_data=(x_test, y_test),
         epochs=100,
         batch_size=32,
         callbacks=callbacks)

preds = model.predict(test_array)
preds_result = [np.argmax(pred) for pred in preds]
# !tensorboard --logdir=./logs




# sample_submission['target'] = clf.predict(test_vectors)
sample_submission['target'] = preds_result
sample_submission.head()
# Submission
sample_submission.to_csv("submission.csv", index=False)
from pprint import pprint

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dot, Flatten, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingModel:
    def __init__(self, vocab_size, emb_dim=100):
        self.word_input = Input(shape=(1,), name='word_input')
        self.word_embed = Embedding(
            input_dim=vocab_size,
            output_dim=emb_dim,
            input_length=1,
            name='word_embedding'
        )
        self.context_input = Input(shape=(1,), name='context_input')
        self.context_embed = Embedding(
            input_dim=vocab_size,
            output_dim=emb_dim,
            input_length=1,
            name='context_embedding'
        )

        self.doc = Dot(axes=2)
        self.flatten = Flatten()
        self.output = Dense(1, activation='sigmoid')

    def build(self):
        word_embed = self.word_embed(self.word_input)
        conetxt_embed = self.context_embed(self.context_input)
        dot = self.doc([word_embed, conetxt_embed])
        flatten = self.flatten(dot)
        output = self.output(flatten)

        model = Model(inputs=[self.word_input, self.context_input], outputs=output)
        return model

    
def create_dataset(text, vocab, num_words, window_size, negative_samples):
    data = vocab.texts_to_sequences([text]).pop()
    sampling = make_sampling_table(num_words)
    couples, labels = skipgrams(
        data, num_words, negative_samples=negative_samples, sampling_table=sampling
    )
    word_target, word_context = zip(*couples)
    word_target = np.reshape(word_target, (-1, 1))
    word_context = np.reshape(word_context, (-1, 1))
    labels = np.asarray(labels)
    return [word_target, word_context], labels

from tensorflow.keras.preprocessing.text import Tokenizer
def build_vocabulary(text, num_words=None):
    tokenizer = Tokenizer(num_words, oov_token='<UNK>')
    tokenizer.fit_on_texts(text)
    return tokenizer


class InferenceAPI:
    def __init__(self, model, vocab, target_layer):
        self.vocab = vocab
        self.weights = model.get_layer(target_layer).get_weights()[0]
        
    def most_similar(self, word, tops=10):
        word_index = self.vocab.word_index.get(word, 1)
        sim = self._cosine_similarity(word_index)
        pairs = [(s, i) for i, s in enumerate(sim)]
        pairs.sort(reverse=True)
        pairs = pairs[1: tops + 1]
        res = [(self.vocab.index_word[i], s) for s, i in pairs]
        return res
        

    def similarity(self, word1, word2):
        word_index1 = self.vocab.word_index.get(word1, 1)
        word_index2 = self.vocab.word_index.get(word2, 1)
        weigh1 = self.weights[word_index1]
        weigh2 = self.weights[word_index2]
        return cosine(weight1, weight2)
    

    def _cosine_similarity(self, target_idx):
        target_weight = self.weights[target_idx]
        similarity = cosine_similarity(self.weights, [target_weight])
        return similarity.flatten()

num_words = 10000
emb_dim = 100
window_size = 1,
negeative_samples = 1

sentences_for_skipgrams = ""
for sentence in train["text"].values:
    sentences_for_skipgrams = sentences_for_skipgrams + sentence + '. '
    

vocab = build_vocabulary([sentences_for_skipgrams], num_words)

x, y = create_dataset(sentences_for_skipgrams, vocab, num_words, window_size, negeative_samples)

model = EmbeddingModel(num_words, emb_dim)
model = model.build()
model.compile(optimizer='adam', loss='binary_crossentropy')

callbacks = [
    EarlyStopping(patience=1),
    ModelCheckpoint("./model.h5", save_best_only=True)
]

model.fit(x, y, batch_size=64, epochs=10, validation_split=0.2, callbacks=callbacks)
# how to get the weight of embedding layer weight
weight = model.get_layer("word_embedding").get_weights()[0]
api = InferenceAPI(model, vocab, "word_embedding")
pprint(api.most_similar(word="breakfast"))
import logging
from gensim.models.word2vec import Word2Vec, Text8Corpus, Text8Corpus
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
train["text"].values[0]
sentences_for_skipgrams = []
for sentence in train["text"].values:
    sentence_list = sentence.split(' ')
    sentences_for_skipgrams.append(sentence_list)
# https://qiita.com/g-k/items/69afa87c73654af49d36
model = Word2Vec(sentences_for_skipgrams, size=1000, window=2, sg=1)
# model.save('model.bin')
# model = Word2Vec.load('model.bin')
# dir(model)
# model.wv.vectors
model.wv.vocab
model.wv['our'].shape
model.wv.most_similar('wow', topn=10)
model.wv.most_similar(positive=['wow', 'friday'], negative=['shower'], topn=100)
model.similarity('game', 'imagin')
# https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
# import gensim
# model = gensim.models.KeyedVectors.load_word2vec_format('/kaggle/input/ccen300vecgz/cc.en.300.vec', binary=False)
# model.most_similar('apple', topn=10)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, SimpleRNN

class RNNModel:
    def __init__(
        self,
        input_dim,
        output_dim,
        emb_dim=300,
        hid_dim=100,
        embeddings=None, # 
        trainable=True # when you setup with weight you can set False not to make it train anymore
    ):
        self.input = Input(shape=(None,), name='input')
        self.embedding = Embedding(input_dim=input_dim, output_dim=output_dim, mask_zero=True, trainable=trainable, name='embedding')
        self.rnn = SimpleRNN(hid_dim, name='rnn')
        self.fc = Dense(output_dim, activation="softmax")
        
    def build(self):
        x = self.input
        embedding = self.embedding(x)
        output = self.rnn(embedding)
        y = self.fc(output)
        return Model(inputs=x, outputs=y)
from tensorflow.keras.preprocessing.text import Tokenizer
def build_vocabulary(text, num_words=None):
    tokenizer = Tokenizer(num_words, oov_token='<UNK>')
    tokenizer.fit_on_texts(text)
    return tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score


def filter_embedding(embeddings, vocab, num_words, dim=300):
    _embeddings = np.zeros((num_words, dim))
    for word in vocab:
        if word in embeddings:
            word_id = vocab[word]
            if word_id >= num_words:
                continue
            _embeddings[word_id] = embeddings[word]
    return _embeddings


def train_model(
    model,
    feats,
    wv = None,
    maxlen: int = 300,
    num_words: int = 40000,
    epochs: int = 100,
    num_label: int = 2
):
    x, y = feats['text'], feats['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    vocab = build_vocabulary(feats['text'], 5000)

    x_train = vocab.texts_to_sequences(x_train)
    x_test = vocab.texts_to_sequences(x_test)

    # https://keras.io/ja/preprocessing/sequence/
    x_train = pad_sequences(x_train, maxlen=maxlen, truncating='post')
    x_test = pad_sequences(x_test, maxlen=maxlen, truncating='post')
    
    if wv:
        wv = filter_embedding(wv, vocab.word_index, num_words)
    
    if model == "rnn":
        model = RNNModel(num_words, num_label, embeddings=None).build()
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['acc'])
        callbacks = [EarlyStopping(patience=5), ModelCheckpoint('rnn_model.h5', save_best_only=True)]
    
    elif model == 'lstm':
        model = LSTMModel(num_words, num_label, embeddings=None).build()
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['acc'])
        callbacks = [EarlyStopping(patience=5), ModelCheckpoint('lstm_model.h5', save_best_only=True)]
    
    elif model == 'cnn':
        model = CNNModel(num_words, num_label, embeddings=None).build()
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['acc'])
        callbacks = [EarlyStopping(patience=5), ModelCheckpoint('cnn_model.h5', save_best_only=True)]

    elif model == 'pre':
        model = CNNModel(num_words, num_label, embeddings=wv).build()
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['acc'])
        callbacks = [EarlyStopping(patience=5), ModelCheckpoint('pre_cnn_model.h5', save_best_only=True)]

    elif model == 'bilstm':
        model = BidirectionalModel(num_words, num_label, embeddings=wv).build()
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['acc'])
        callbacks = [EarlyStopping(patience=5), ModelCheckpoint('bilstm_model.h5', save_best_only=True)]

    else:
        raise ValueError("no such model defined")
    
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
    )
    
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, -1)
    print(f1_score(y_pred, y_test))
    print(precision_score(y_pred, y_test))
    print(recall_score(y_pred, y_test))
    
    return model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model


class LSTMModel:
    def __init__(
        self,
        input_dim,
        output_dim,
        emb_dim=300,
        hid_dim=100,
        embeddings=None,
        trainable=True
    ):
        self.input = Input(shape=(None,), name='input')
        
        if embeddings is None:
            self.embedding = Embedding(
                input_dim=input_dim,
                output_dim=emb_dim,
                mask_zero=True,
                trainable=trainable,
                name='embedding'
            )

        else:
            self.embedding = Embedding(
                input_dim=embeddings.shape[0],
                output_dim=embeddings.shape[1],
                mask_zero=True,
                trainable=trainable,
                name='embedding'
            )
        self.lstm = LSTM(hid_dim, name='lstm')
        self.fc = Dense(output_dim, activation='softmax')
        
    def build(self):
        x = self.input
        embedding = self.embedding(x)
        lstm = self.lstm(embedding)
        output = self.fc(lstm)
        return Model(inputs=x, outputs=output)
trained_model = train_model(model='lstm', feats=train)
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model

class CNNModel:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: int = 250,
        kernel_size: int = 3,
        emb_dim: int = 300,
        embeddings=None,
        trainable: bool = True
    ):
        self.input = Input(shape=(None,), name='input')
        if embeddings is not None:
            self.embedding = Embedding(
                input_dim=input_dim,
                output_dim=emb_dim,
                trainable=trainable,
                name='embedding'
            )
        else:
            self.embedding = Embedding(
                input_dim=embeddings.shape[0],
                ouput_dim=embeddings.shape[1],
                trainable=trainable,
                name='embedding'
            )
        self.conv = Conv1D(
            filters,
            kernel_size,
            padding='valid',
            activation='relu',
            strides=1
        )
        self.pool = GlobalMaxPooling1D()
        self.fc = Dense(output_dim, activation='softmax')
        
    
    def build(self):
        x = self.input
        embedding = self.embedding(x)
        conv = self.conv(embedding)
        pool = self.pool(conv)
        y = self.fc(pool)
        return Model(inputs=x, outputs=y)
trained_model = train_model(model='cnn', feats=train)
# loading time may take a quite long time
import gensim
wv = gensim.models.KeyedVectors.load_word2vec_format('/kaggle/input/ccen300vecgz/cc.en.300.vec', binary=False)
trained_model = train_model(model='pre', feats=train, wv=wv)
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM
from tensorflow.keras.layers import Bidirectional


class BidirectionalModel:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        emb_dim: int = 300,
        hid_dim: int = 100,
        embeddings=None,
        trainable=True
    ):
        self.input = Input(shape=(None,), name='input')
        self.embedding = Embedding(
            input_dim=input_dim,
            output_dim=emb_dim,
            mask_zero=True,
            trainable=trainable,
            name='embedding'
        )
        lstm = LSTM(
            hid_dim,
            return_sequences=False,
            name='lstm'
        )
        self.bilstm = Bidirectional(lstm, name='bilstm')
        self.fc = Dense(output_dim, activation='softmax')
        
    def build(self):
        x = self.input
        embedding = self.embedding(x)
        lstm = self.bilstm(embedding)
        y = self.fc(lstm)
        return Model(inputs=x, outputs=y)
trained_model = train_model(model='bilstm', feats=train,)
!pip install keras_bert
# from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths

# model_path = get_pretrained(PretrainedList.multi_cased_base)
# paths = get_checkpoint_paths(model_path)
# print(paths.config, paths.checkpoint, paths.vocab)


# https://trafalbad.hatenadiary.jp/entry/2019/07/20/085725
!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip -o uncased_L-12_H-768_A-12.zip
!ls ./uncased_L-12_H-768_A-12
# https://trafalbad.hatenadiary.jp/entry/2019/07/20/085725
import os

SEQ_LEN = 128
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-4

pretrained_path = 'uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

import codecs
from keras_bert import load_trained_model_from_checkpoint

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)
from keras_bert import Tokenizer

tokenizer = Tokenizer(token_dict)
from abc import ABCMeta, abstractmethod
import tensorflow as tf


class BaseModel(metaclass=ABCMeta):
    def __init__(
        self,
        use_bn: bool = False,
        dropout_rate: float = 0.2,
        hidden_unit: int = 356,
        num_layers: int = 2,
        task_type: str = "classification"
    ):
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.hidden_unit = hidden_unit
        self.num_layers = num_layers
        self.task_type = task_type

    def mlp(self, x: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        for i in range(2):
            x = tf.keras.layers.Dense(
                self.hidden_unit, activation=tf.keras.activations.relu, name=f"embedding_textual" if i == 0 else None
            )(x)
            if self.use_bn:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            self.hidden_unit //= 2
        return tf.keras.layers.Dense(
            self.num_layers,
#             activation="linear" if self.task_type == "regression" else "softmax",
            activation="linear",
            name="output_context",
        )(x)

    @abstractmethod
    def build(self) -> tf.keras.Model:
        raise NotImplementedError
import tensorflow as tf
from keras_bert import AdamWarmup, calc_train_steps, load_trained_model_from_checkpoint


class TextualModel(BaseModel):
    def __init__(
        self,
        input_shape: int = 128,
        layer_name: str = "x_textual",
        trainable: bool = True,
        multi_inputs: bool = False,
        lr: float = 1e-3,
        loss: str = "mse",
        metrics: str = "mae",

        use_bn: bool = False,
        dropout_rate: float = 0.2,
        hidden_unit: int = 356,
        num_layers: int = 2,
        task_type: str = "classification"
    ):
        self.input_shape = input_shape
        self.layer_name = layer_name
        self.trainable = trainable
        self.multi_inputs = multi_inputs
        self.lr = lr
        self.loss = loss
        self.metrics = metrics
        super().__init__(
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            hidden_unit=hidden_unit,
            num_layers=num_layers,
            task_type=task_type
        )

    def build(self) -> tf.keras.Model:
        if self.multi_inputs:
            model_suffix = "_multi"
            # [CLS] + len(outside_tokens) + [SEP] + len(inside_tokens) + [SEP]
            seq_len = 1 + (self.input_shape - 2 + 1) * 2
        else:
            model_suffix = ""
            seq_len = self.input_shape
        bert = load_trained_model_from_checkpoint(
            config_path,
            checkpoint_path,
            training=True,
            trainable=True,
            seq_len=SEQ_LEN,
        )
        # Rename input layer
        bert.layers[0]._name = f"x_textual_token"
        bert.layers[1]._name = f"x_textual_segment"
        inputs = (bert.inputs[0], bert.inputs[1])
        if self.trainable:
            model_name = f"bert_finetuned{model_suffix}"
            x = bert.get_layer("NSP-Dense").output
        else:
            model_name = f"bert_vanilla{model_suffix}"
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(356))(bert.output)

        outputs = self.mlp(x)
        model = tf.keras.Model(inputs, outputs, name=model_name)

        decay_steps, warmup_steps = calc_train_steps(
            num_example=seq_len, batch_size=64, epochs=50
        )
#         model.compile(
#             # loss="mse" if self.task_type == "regression" else "categorical_crossentropy",
#             loss="categorical_crossentropy",
#             optimizer=AdamWarmup(decay_steps=decay_steps,
#                                  warmup_steps=warmup_steps, min_lr=1e-4),
#             # metrics=["mae"] if self.task_type == "regression" else ["acc", "mse", "mae"],
#             metrics=["acc"]
#         )
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['acc'])
        return model

BertClasificationModel = TextualModel().build()
BertClasificationModel.summary()
from keras.utils import plot_model
plot_model(BertClasificationModel, to_file='model.png')
train['text']
# x_train, y_train
from keras.utils import np_utils

def make_token_seg(x, y):
    indices, sentiments = [], []
    for token, tar in zip(train['text'], train['target']):
        ids, segments = tokenizer.encode(token, max_len=SEQ_LEN)
        indices.append(ids)
        sentiments.append(tar)
    indices = np.array(indices)
    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    x = [indices, np.zeros_like(indices)]
    y = np.array(sentiments)
    return x, np_utils.to_categorical(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train["text"], train["target"], test_size=0.2, random_state=10)
x_train, y_train = make_token_seg(x_train, y_train)
x_test, y_test = make_token_seg(x_test, y_test)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score
callbacks = [EarlyStopping(patience=5), ModelCheckpoint('bert_classification.h5', save_best_only=True)]


BertClasificationModel.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    callbacks=callbacks,
    shuffle=True,
)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, -1)
print(f1_score(y_pred, y_test))
print(precision_score(y_pred, y_test))
print(recall_score(y_pred, y_test))

return model
result = BertClasificationModel.predict(x, verbose=True).argmax(-1)
print(result)
a = [1, 2, 3]
a[:2]
