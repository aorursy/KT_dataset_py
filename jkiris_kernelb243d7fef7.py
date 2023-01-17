import os
import re
import sys
import pickle
import time
import math
import random
import datetime
import shutil
import textblob

import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.training import HParams
from nltk.tokenize import TweetTokenizer
from tensorflow.contrib.learn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
_WORD_RE = dict()

def _get_regexp(pat):
    if pat not in _WORD_RE:
        _WORD_RE[pat] = re.compile(pat)
    return _WORD_RE[pat]

def word_cut(sentence, cut_by=' '):
    # words = filter(bool, str_decode(sentence, coding).split(' '))

    words = filter(bool, _get_regexp(cut_by).split(sentence))
    return words


class WordVocabProcessor(preprocessing.VocabularyProcessor):
    def __init__(self, max_document_length, min_frequency=0,
                 vocabulary=None, cut_by=' ', **kwargs):
        super(WordVocabProcessor, self).__init__(max_document_length, min_frequency,
                                                  vocabulary, tokenizer_fn=None)
        self.cut_by = cut_by

    def fit(self, raw_documents, unused_y=None):
        self.vocabulary_.add('<PAD>')
        for tokens in self.tokenizer_(raw_documents):
            for token in tokens:
                self.vocabulary_.add(token)

        if self.min_frequency > 0:
            self.vocabulary_.trim(self.min_frequency)

        self.vocabulary_.freeze()
        return self

    def tokenizer_(self, documents):
        for doc in documents:
            yield word_cut(doc, cut_by=self.cut_by)

    def transform(self, raw_documents):
        for tokens in self.tokenizer_(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)

            yield word_ids

    @property
    def vocab_size(self):
        return len(self.vocabulary_)

def load_train_df():
    df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', '\t')
#     df['Phrase'] = df['Phrase'].str.lower()
    return df

def load_test_df():
    df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', '\t')
#     df['Phrase'] = df['Phrase'].str.lower()
    return df

def load_made_df():
    return pd.concat([pd.read_csv('../input/movie-review-sentiment-analysis-man-made/trans_0.csv', ','), 
                         pd.read_csv('../input/movie-review-sentiment-analysis-man-made/trans_1.csv', ',')])

def build_word_vocab():
    df_train = load_train_df()
    df_test = load_test_df()
    texts = list(df_train.Phrase) + list(df_test.Phrase)
    
    vocab = WordVocabProcessor(max_document_length=60, cut_by=r'[ \-\\/]')
    vocab.fit(texts)
    print('Build word vocab with size: %s' % vocab.vocab_size)
    return vocab
class UniformRandomEmbedding(object):
    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.W_ = None

    def W(self):
        if self.W_ is None:
            self.W_ = np.random.uniform(-1.0, 1.0, (self.vocab_size, self.embedding_size))

        return self.W_.astype("float32")

    def save(self, filename):
        if self.W_ is None:
            self.W()

        with open(filename, 'wb') as f:
            f.write(pickle.dumps(self))

    @classmethod
    def restore(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.loads(f.read())

        
class FastTextEmbedding(UniformRandomEmbedding):
    def __init__(self, vocab):
        self.vocab = vocab
        super(FastTextEmbedding, self).__init__(vocab.vocab_size, 300 + 2)

    def W(self):
        vocab_size = self.vocab.vocab_size
        
        if self.W_ is None:
            words_need = set([self.vocab.vocabulary_.reverse(idx) for idx in range(vocab_size)])
            words_need |= set([w.lower() for w in words_need])
            words_need.add("n't")
            
            word_vec = dict()
            with open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec') as f:
                f.readline()
                for line in f:
                    dd = line.strip().split(' ')
                    word, vec = dd[0], np.asarray(dd[1:], dtype="float32")
                    if word in words_need:
                        word_vec[word] = vec

            W = super(FastTextEmbedding, self).W()
            exists_cnt = 0
            for idx in range(vocab_size):
                word = self.vocab.vocabulary_.reverse(idx)
                sent = textblob.TextBlob(word).sentiment
                W[idx, -2:] = [sent.polarity, sent.subjectivity]

                if word == "n't":
                    word = 'not'
                    print('redirect n\'t to not')

                if word in word_vec:
                    exists_cnt += 1
                    W[idx, :-2] = word_vec[word]
                elif word.lower() in word_vec:
                    exists_cnt += 1
                    W[idx, :-2] = word_vec[word.lower()]
                else:
                    print("Can not find %s in fast text embedding." % word)
            print('Found %s / %s in fast-text embedding vocab.' % (exists_cnt, vocab_size))

            del word_vec

            self.W_ = W

        return self.W_.astype("float32")

    
def build_fast_text_embedding(vocab):
    embedding = FastTextEmbedding(vocab)
    embedding.W()
    return embedding


class GloveEmbedding(UniformRandomEmbedding):
    def __init__(self, vocab, embedding_size=300):
        assert embedding_size in (50, 100, 200, 300)
        self.vocab = vocab
        self.embedding_size_raw = embedding_size
        super(GloveEmbedding, self).__init__(vocab.vocab_size, embedding_size + 2)

    def W(self):
        vocab_size = self.vocab.vocab_size
        
        if self.W_ is None:
            words_need = set([self.vocab.vocabulary_.reverse(idx) for idx in range(vocab_size)])
            words_need |= set([w.lower() for w in words_need])
            words_need.add("n't")
            
            word_vec = dict()
            with open('../input/glove6b/glove.6B.%sd.txt' % self.embedding_size_raw) as f:
                for line in f:
                    dd = line.strip().split(' ')
                    word, vec = dd[0], np.array(list(map(float, dd[1:])))
                    if word in words_need:
                        word_vec[word] = vec

            W = super(GloveEmbedding, self).W()
            exists_cnt = 0
            for idx in range(vocab_size):
                word = self.vocab.vocabulary_.reverse(idx)
                sent = textblob.TextBlob(word).sentiment
                W[idx, -2:] = [sent.polarity, sent.subjectivity]
                
                if word in word_vec:
                    exists_cnt += 1
                    W[idx, :-2] = word_vec[word]
                elif word.lower() in word_vec:
                    exists_cnt += 1
                    W[idx, :-2] = word_vec[word.lower()]
                else:
                    print("Can not find %s in glove embedding." % word)
            print('Found %s / %s in glove embedding vocab.' % (exists_cnt, vocab_size))

            del word_vec

            self.W_ = W

        return self.W_.astype("float32")
    

def build_glove_embedding(vocab):
    embedding = GloveEmbedding(vocab, 100)
    embedding.W()
    return embedding
class LabelEncoder(object):
    def __init__(self):
        self.labels_ = None

    def _check_fit(self):
        assert self.labels_, 'LabelEncode not fit yet.'

    def fit(self, y):
        self.labels_ = list(set(y))
        return self

    def transform(self, y):
        self._check_fit()

        labels_d = dict([(l, i) for i, l in enumerate(self.labels_)])

        labels_m = np.zeros((len(y), len(self.labels_)))
        for idx, label in enumerate(y):
            labels_m[idx, labels_d.get(label)] = 1.

        return labels_m

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def reverse(self, logits):
        self._check_fit()

        return [self.labels_[idx] for idx in np.argmax(logits, 1)]

    @property
    def num_classes(self):
        self._check_fit()

        return len(self.labels_)

    def save(self, filename):
        with open(filename, 'wb') as f:
            f.write(pickle.dumps(self))

    @classmethod
    def restore(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.loads(f.read())

def build_label_encoder():
    df_train = load_train_df()
    labels = list(df_train.Sentiment)
    
    label_ed = LabelEncoder()
    label_ed.fit(labels)
    print('Encode %s classes.' % label_ed.num_classes)

    return label_ed


class BinaryLabelEncoder(LabelEncoder):
    def __init__(self):
        self.labels_ = [0, 2]

    def _check_fit(self):
        assert self.labels_, 'LabelEncode not fit yet.'

    def fit(self, y):
        return self

    def transform(self, y):
        self._check_fit()

        labels_m = np.zeros((len(y), len(self.labels_)))
        for idx, label in enumerate(y):
            if label == 2:
                labels_m[idx, 1] = 1.

        return labels_m


def build_binary_label_encoder():
    label_ed = BinaryLabelEncoder()
    return label_ed

vocab = build_word_vocab()
# embedding_g = build_glove_embedding(vocab)
embedding_ft = build_fast_text_embedding(vocab)
label_ed = build_label_encoder()
def computeTF(df):
    dicts = []
    for i in range(0, 5):
        tf_dict = {}
        sentences = list(df[df['Sentiment']==i]['Phrase'].values)
        for sentence in sentences:
            sentence = sentence.lower()
            words = sentence.split()
            for word in words:
                if word not in tf_dict:
                    tf_dict[word] = 1
                else:
                    tf_dict[word] += 1
        total_words = sum(tf_dict.values())
        for word, val in tf_dict.items():
            tf_dict[word] = val * 1.0/total_words
        dicts.append(tf_dict)
    return dicts

def computeIDF(tf_dicts):
    keys_all = []
    idf_dict = {}
    for i in range(0, 5):
        keys_all += list(tf_dicts[i].keys())
    for key in keys_all:
        if key not in idf_dict:
            idf_dict[key] = 1
        else:
            idf_dict[key] += 1
    for word, val in idf_dict.items():
            idf_dict[word] = math.log(5.0 / idf_dict[word])
    return idf_dict

def computeTFIDF(tf_dicts, idf_dict):
    tfidf_dicts = []
    for i in range(0, 5):
        tfidf_dict = {}
        for word, val in tf_dicts[i].items():
            tfidf_dict[word] = tf_dicts[i][word] * idf_dict[word]
        tfidf_dicts.append(tfidf_dict)
    return tfidf_dicts

def prepare_keywords():
    df = load_train_df()
    tf_dicts = computeTF(df)
    idf_dict = computeIDF(tf_dicts)
    tfidf_dicts = computeTFIDF(tf_dicts, idf_dict)

    keywords = []
    for i in range(0, 5):
        important_words = sorted(tfidf_dicts[i].items(), key=lambda x: x[1], reverse=True)[1:100]
        keywords.append([item[0] for item in important_words])
    all_keywords = [keyword for keyword_list in keywords for keyword in keyword_list]
    
    return keywords, all_keywords

keywords, all_keywords = prepare_keywords()
class Node(object):
    def __init__(self, parent, pid, phrase):
        self.parent = parent
        self.children = []
        self.pid = pid
        self.phrase = phrase
    
    def add_child(self, node):
        self.children.append(node)

def build_sentence_tree(dd):
    root, current = None, None
    
    for _, row in dd.iterrows():
        pid, phrase = row.PhraseId, row.Phrase
        if not root:
            root = Node(None, pid, phrase)
            current = root
            continue
        
        while True:
            if current is None:
                break
            if phrase in current.phrase:
                node = Node(current, pid, phrase)
                current.add_child(node)
                current = node
                break
            else:
                current = current.parent
                
    pids_start, current = [], root
    while current:
        pids_start.append(current.pid)
        if not current.children:
            break
        current = current.children[0]
    
    pids_end, current = [], root
    while current:
        pids_end.append(current.pid)
        if not current.children:
            break
        current = current.children[-1]
    
    pid_level_dct = dict()
    level, lid, pids_leaf = [root], 1, []
    while level:
        pids_leaf += [n.pid for n in level if not n.children]
        for n in level:
            pid_level_dct[n.pid] = lid
        
        level = [n for r in level for n in r.children]
        lid += 1
        
    return pd.DataFrame([(pid,
                          1 if pid in pids_start else 0, 
                          1 if pid in pids_end else 0, 
                          1 if pid in pids_leaf else 0, 
                          pid_level_dct.get(pid, lid)) for pid in dd.PhraseId], 
                        columns=['PhraseId', 'is_start', 'is_end', 'is_leaf', 'tree_level'], 
                        index=dd.index)

def tree_feat(df):
    tree_feat = pd.concat([build_sentence_tree(df[df.SentenceId == sid]) for sid in df.SentenceId.unique()])
    assert tree_feat.shape[0] == df.shape[0]
    return tree_feat  
FEATS_DENCE = ["phrase_count", "word_count", "has_upper", "sentence_end", "after_comma", 
              "sentence_start", 'is_start', 'is_end', 'is_leaf', 'tree_level', 
              "sentiment0_words", "sentiment1_words","sentiment3_words", "sentiment4_words", "no_sentiment_words"]

def transform_dence(df):
    dence_df = pd.DataFrame({
        "PhraseId": df.PhraseId,
        "phrase_count": df.groupby("SentenceId")["Phrase"].transform("count"),
        "word_count": df["Phrase"].apply(lambda x: len(x.split())),
        "has_upper": df["Phrase"].apply(lambda x: x.lower() != x),
        "sentence_end": df["Phrase"].apply(lambda x: x.endswith(".")),
        "after_comma": df["Phrase"].apply(lambda x: x.startswith(",")),
        "sentence_start": df["Phrase"].apply(lambda x: "A" <= x[0] <= "Z"),
        "sentiment0_words": df["Phrase"].apply(lambda x: len(set(x.split()).intersection(set(keywords[0])))),
        "sentiment1_words": df["Phrase"].apply(lambda x: len(set(x.split()).intersection(set(keywords[1])))),
        "sentiment3_words": df["Phrase"].apply(lambda x: len(set(x.split()).intersection(set(keywords[3])))),
        "sentiment4_words": df["Phrase"].apply(lambda x: len(set(x.split()).intersection(set(keywords[4])))),
        "no_sentiment_words": df["Phrase"].apply(lambda x: len(set(x.split()))-len(set(x.split()).intersection(set(all_keywords)))),
    })
    
    tree_df = tree_feat(df)  
    
    df_d = pd.merge(dence_df, tree_df, on="PhraseId")
    assert df_d.shape[0] == df.shape[0]
    return df_d.loc[:, FEATS_DENCE]


def prepare_train(df, random_state=None):
    df_d = transform_dence(df)
    
    df = pd.concat([df, df_d], axis=1)
    
    if random_state is None:
        random_state = random.randint(0, 1000)

    print('Shuffle with random_state: %s' % random_state)
    
    df = df.sample(frac=1, random_state=random_state)
    
    return np.array(df.Phrase), np.array(df.SentenceId), np.array(df.Sentiment), np.array(df.loc[:, FEATS_DENCE])
class TrainInputFeeder(object):
    def __init__(self, X, y, vocab, label_ed=None, num_epochs=5, batch_size=128, shuffle=True):
        self.X = np.array(X)
        self.y = np.array(y)

        self.vocab = vocab
        self.label_ed = label_ed
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle

    def sample_weight(self, y):
        y = np.array(y)
        return np.ones((y.shape[0], 1))
    
    def transform_X(self, X):
        return np.array(list(self.vocab.transform(X)))
    
    def transform_y(self, y):
        if self.label_ed is None:
            return y
        
        return self.label_ed.transform(y)
    
    def __iter__(self):
        X_size = len(self.X)
        num_batches_per_epoch = int((X_size - 1) / self.batch_size) + 1
        print('num batches per epoch: %s' % num_batches_per_epoch)

        for epoch in range(self.num_epochs):
            print
            print('start epoch %s' % (epoch + 1))
            print

            if self.shuffle:
                shuffle_indices = np.random.permutation(np.arange(X_size))
                X, y = self.X[shuffle_indices], self.y[shuffle_indices]
            else:
                X, y = self.X, self.y

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, X_size)
                yield (self.transform_X(X[start_index:end_index]), 
                        self.transform_y(y[start_index:end_index]))

                
class TrainInputFeeder2X(object):
    def __init__(self, X1, X2, y, num_epochs=5, batch_size=128, shuffle=True):
        self.X1 = np.array(X1)
        self.X2 = np.array(X2)
        self.y = np.array(y)

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle

    def sample_weight(self, y):
        y = np.array(y)
        return np.ones((y.shape[0], 1))
    
    def __iter__(self):
        X_size = len(self.X1)
        num_batches_per_epoch = int((X_size - 1) / self.batch_size) + 1
        print('num batches per epoch: %s' % num_batches_per_epoch)

        for epoch in range(self.num_epochs):
            print
            print('start epoch %s' % (epoch + 1))
            print

            if self.shuffle:
                shuffle_indices = np.random.permutation(np.arange(X_size))
                X1, X2, y = self.X1[shuffle_indices], self.X2[shuffle_indices], self.y[shuffle_indices]
            else:
                X1, X2, y = self.X1, self.X2, self.y

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, X_size)
                yield (X1[start_index:end_index], X2[start_index:end_index], 
                        y[start_index:end_index])
                

class PredictInputFeeder2X(object):
    def __init__(self, X1, X2, batch_size=128):
        self.X1 = np.array(X1)
        self.X2 = np.array(X2)
        self.result = []

        self.batch_size = batch_size

    def __iter__(self, merge=True):
        X_size = len(self.X1)
        num_batches = int((X_size - 1) / self.batch_size) + 1

        for batch_num in range(num_batches):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, X_size)
            yield self.X1[start_index:end_index], self.X2[start_index:end_index]

    def feed_result(self, result):
        self.result.extend(result)

class PredictInputFeeder(object):
    def __init__(self, X, vocab, batch_size=128):
        self.X = np.array(X)
        self.result = []

        self.vocab = vocab
        self.batch_size = batch_size
    
    def transform_X(self, X):
        return np.array(list(self.vocab.transform(X)))

    def __iter__(self, merge=True):
        X_size = len(self.X)
        num_batches = int((X_size - 1) / self.batch_size) + 1

        for batch_num in range(num_batches):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, X_size)
            yield self.transform_X(self.X[start_index:end_index])

    def feed_result(self, result):
        self.result.extend(result)
def _single_cell(unit_type, num_units):
    """Create an instance of a single RNN cell."""

    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "nas":
        single_cell = tf.contrib.rnn.NASCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    return single_cell


def build_cell(unit_type, num_units, num_layers):
    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(
            unit_type=unit_type,
            num_units=num_units,
        )
        cell_list.append(single_cell)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)

    
def batch_norm(x, variance_epsilon=0.001):
    shape_x = x.get_shape()
    with tf.name_scope("bn"):
        axis = list(range(len(shape_x) - 1))
        wb_mean, wb_var = tf.nn.moments(x, axis)
        scale = tf.Variable(tf.ones(shape_x[-1:]))
        offset = tf.Variable(tf.zeros(shape_x[-1:]))

        h_bn = tf.nn.batch_normalization(x, wb_mean, wb_var, offset, scale, variance_epsilon)

    return h_bn


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


class TextRnnCnn(object):
    def __init__(self, sequence_length, num_classes, embedding_W,
                 num_units, attention_size, filter_sizes, num_filters, 
                 num_outputs_fc1, num_outputs_fc2, num_outputs_fc3):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_d = tf.placeholder(tf.float32, [None, len(FEATS_DENCE)], name="input_d")

        embedding_inputs = self.embed(embedding_W)
        seq_length = self._seq_length(self.input_x)

        self.embedding_keep_prob = tf.placeholder(tf.float32, name="embedding_keep_prob")
        self.embedding_inputs = tf.nn.dropout(embedding_inputs, self.embedding_keep_prob)

        rnn_output_lstm, rnn_output_lstm_last = self.build_rnn(self.embedding_inputs, 'lstm', num_units, 
                                                               seq_length, variable_scope='lstm')
        rnn_output_lstm_attention, _ = attention(rnn_output_lstm, attention_size, return_alphas=True)
        cnn_output_lstm = self.build_cnn(rnn_output_lstm, sequence_length, filter_sizes, num_filters,
                                         variable_scope='lstm-cnn')

        rnn_output_gru, rnn_output_gru_last = self.build_rnn(self.embedding_inputs, 'gru', num_units, 
                                                             seq_length, variable_scope='gru')
        rnn_output_gru_attention, _ = attention(rnn_output_gru, attention_size, return_alphas=True)
        cnn_output_gru = self.build_cnn(rnn_output_gru, sequence_length, filter_sizes, num_filters,
                                         variable_scope='gru-cnn')

        self.fc_keep_prob = tf.placeholder(tf.float32, name="fc_keep_prob")

#         cnn_output = tf.concat([cnn_output_lstm, cnn_output_gru], 1)
        cnn_output = tf.concat([cnn_output_lstm, cnn_output_gru, self.input_d], 1)
#         cnn_output = tf.concat([rnn_output_lstm_attention, rnn_output_gru_attention, self.input_d], 1)
        self.cnn_output = tf.nn.dropout(cnn_output, self.fc_keep_prob)

        with tf.name_scope("fc1"):
            h_fc1_bn = batch_norm(self.cnn_output)
            print("Level fc1 shape: %s" % h_fc1_bn.get_shape()[1])
            h_fc1 = tf.contrib.layers.fully_connected(h_fc1_bn, num_outputs_fc1, activation_fn=tf.nn.relu)
            h_fc1 = tf.concat([h_fc1, self.input_d], 1)
            h_fc2_d = tf.nn.dropout(h_fc1, self.fc_keep_prob)
        
#         with tf.name_scope("fc2"):
#             h_fc2_bn = batch_norm(h_fc1_d)
#             print("Level fc2 shape: %s" % h_fc2_bn.get_shape()[1])
#             h_fc2 = tf.contrib.layers.fully_connected(h_fc2_bn, num_outputs_fc2, activation_fn=tf.nn.relu)
#             h_fc2 = tf.concat([h_fc2, self.input_d], 1)
#             h_fc2_d = tf.nn.dropout(h_fc2, self.fc_keep_prob)

        with tf.name_scope("fc3"):
            h_fc3_bn = batch_norm(h_fc2_d)
            print("Level fc3 shape: %s" % h_fc3_bn.get_shape()[1])
            self.h_out = tf.contrib.layers.fully_connected(h_fc3_bn, num_outputs_fc3, activation_fn=tf.nn.relu)
            self.h_out = tf.concat([self.h_out, self.input_d], 1)
            # self.h_out = tf.nn.dropout(h_fc2, self.fc_keep_prob)

        # Final (unnormalized) logits and predictions
        with tf.name_scope("output"):
            self.h_out = batch_norm(self.h_out)
            print("Level output shape: %s" % self.h_out.get_shape()[1])
            self.logits = tf.contrib.layers.fully_connected(self.h_out, num_classes)
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def embed(self, embedding_W):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(embedding_W, name="W")
            embedding_inputs = tf.nn.embedding_lookup(self.W, self.input_x)

        return embedding_inputs

    def build_cnn(self, rnn_outputs, sequence_length, filter_sizes, num_filters, variable_scope='cnn'):
        rnn_outputs = tf.expand_dims(rnn_outputs, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("%s-conv-%s" % (variable_scope, filter_size),
                                   initializer=tf.keras.initializers.he_uniform()):
                # Convolution Layer
                filter_shape = [filter_size, int(rnn_outputs.get_shape()[2]), 1, num_filters]
                W = tf.get_variable("W", shape=filter_shape)
                b = tf.get_variable("b", shape=[num_filters])
                conv = tf.nn.conv2d(
                    rnn_outputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled_max = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled_max)

                pooled_avg = tf.nn.avg_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled_avg)

        # Combine all the pooled features
        num_filters_total = num_filters * len(pooled_outputs)
        h_pool = tf.concat(pooled_outputs, 3)

        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        return h_pool_flat

    def build_rnn(self, embedding_inputs, unit_type, num_units, seq_length, variable_scope="rnn"):
        with tf.variable_scope(variable_scope):#, initializer=tf.orthogonal_initializer()):
            cell_fw = build_cell(unit_type, num_units, 1)
            cell_bw = build_cell(unit_type, num_units, 1)

            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, embedding_inputs,
                dtype=tf.float32, sequence_length=seq_length,
                swap_memory=True)

            rnn_outputs = tf.concat(bi_outputs, 2)

#             cell = build_cell(unit_type, num_units, 1)
#             rnn_outputs, state = tf.nn.dynamic_rnn(
#                 cell, embedding_inputs, dtype=tf.float32, sequence_length=seq_length,
#                 swap_memory=True)

        return rnn_outputs, self.last_relevant(rnn_outputs, seq_length)

    @staticmethod
    def _seq_length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + tf.maximum(length - 1, 0)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)

    
def build_model(hparams, vocab, label_ed, embedding_W):
    return TextRnnCnn(
            vocab.max_document_length, label_ed.num_classes, embedding_W,
            hparams.num_units, hparams.attention_size, 
            hparams.filter_sizes, hparams.num_filters,
            hparams.num_outputs_fc1, hparams.num_outputs_fc2, hparams.num_outputs_fc3
    )


class EarlyStopError(Exception):
    pass


class EarlyStopMonitor(object):
    BEST_INIT_VALUE = -999.0

    def __init__(self, sess, saver, mode='min', min_delta=0, patience=0, baseline=None):
        self.sess = sess
        self.saver = saver

        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta

        if mode not in ['min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        self.wait = 0
        self.best = None
        if self.get_best() is None and self.baseline is not None:
            self.set_best(self.baseline)

    def get_best(self):
        return self.best

    def set_best(self, value):
        self.best = value

    def reset_wait(self):
        self.wait = 0

    def get_wait(self):
        return self.wait

    def incr_wait(self):
        self.wait += 1

    def save_checkpoint(self, *args, **kwargs):
        path = self.saver.save(self.sess, *args, **kwargs)
        print("Saved model checkpoint to {}\n".format(path))

    def on_eval(self, current, *args, **kwargs):
        best = self.get_best()
        if best is None:
            self.set_best(current)
            self.reset_wait()
            self.save_checkpoint(*args, **kwargs)
            return

        if self.monitor_op(current - self.min_delta, best):
            self.set_best(current)
            self.save_checkpoint(*args, **kwargs)
            self.reset_wait()
        else:
            self.incr_wait()
            if self.get_wait() >= self.patience:
                raise EarlyStopError()
def train(hparams, feeder, X_dev, d_dev, y_dev, embedding_W,
          num_checkpoints=3, evaluate_every=200, patience=5, 
          out_dir=None):
    timestamp = str(int(time.time()))
    if out_dir is None:
        name = '%s_%s' % (hparams.out_prefix, timestamp)
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", name))
    
    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            # Define Training procedure
            rnn = build_model(hparams, vocab, label_ed, embedding_W)

            global_step = tf.Variable(0, name="global_step", trainable=False)

            learning_rate = hparams.learning_rate
            learning_rate_t = tf.placeholder(tf.float32, name="learning_rate")
            optimizer = tf.train.AdamOptimizer(learning_rate_t)
            grads_and_vars = optimizer.compute_gradients(rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            print("Writing to {}\n".format(out_dir))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
            monitor = EarlyStopMonitor(sess, saver, mode='max', patience=patience, min_delta=0.)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                # Initialize all variables
                sess.run(tf.global_variables_initializer())

            def train_step(x_batch, d_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    rnn.input_x: x_batch,
                    rnn.input_y: y_batch,
                    rnn.input_d: d_batch,
                    rnn.embedding_keep_prob: hparams.embedding_keep_prob,
                    rnn.fc_keep_prob: hparams.fc_keep_prob,
                    learning_rate_t: learning_rate,
                }

                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, rnn.loss, rnn.accuracy],
                    feed_dict)

                if hparams.verbose > 1:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, d_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    rnn.input_x: x_batch,
                    rnn.input_y: y_batch,
                    rnn.input_d: d_batch,
                    rnn.embedding_keep_prob: 1.0,
                    rnn.fc_keep_prob: 1.0,
                }
                step, loss, accuracy = sess.run(
                    [global_step, rnn.loss, rnn.accuracy],
                    feed_dict)
                if hparams.verbose > 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                return accuracy

            for X_batch, d_batch, y_batch in feeder:
                train_step(X_batch, d_batch, y_batch)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    accuracy = dev_step(X_dev, d_dev, y_dev)
                    print("")

                    try:
                        monitor.on_eval(accuracy, checkpoint_prefix, global_step=current_step)
                    except EarlyStopError as e:
                        break

    return out_dir
def predict(hparams, feeder, out_dir, vocab, label_ed, embedding_W):
    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            # Define Training procedure
            rnn = build_model(hparams, vocab, label_ed, embedding_W)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            writer = tf.summary.FileWriter(out_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            for X_batch, d_batch in feeder:
                feed_dict = {
                    rnn.input_x: X_batch,
                    rnn.input_d: d_batch,
                    rnn.embedding_keep_prob: 1.0,
                    rnn.fc_keep_prob: 1.0,
                }

                logits = sess.run(rnn.logits, feed_dict)
                feeder.feed_result(logits)
    
    return feeder.result

def train_predict_cv(hparams, feeder_train, texts_dev, label_dev, dence_dev, feeder_pred):
    assert hparams.embed_type in ('fast-text', 'glove')

    if hparams.embed_type == 'fast-text':
        print('Train Use Fast Text embedding.')
        embedding_W = embedding_ft.W() 
    else:
        print('Train Use Glove embedding.')
        embedding_W =  embedding_g.W()
    
    if not hparams.out_dir:
        hparams.out_dir = train(hparams, feeder_train, texts_dev, dence_dev, label_dev, embedding_W, 
                                evaluate_every=hparams.evaluate_every, patience=3, 
                                out_dir=None)

    return np.array(predict(hparams, feeder_pred, hparams.out_dir, vocab, label_ed, embedding_W))


def kfold_by_sentence(sid, n_splits):
    for i in range(n_splits):
        dev_index = sid % n_splits == i
        train_index = (~dev_index)
        yield train_index, dev_index


def blend_cv(hparams, n_splits, path_predict):
    df_train = load_train_df()
    df_test = load_test_df()
    
    df_train['PhraseLower'] = df_train["Phrase"].str.lower()
    df_test['PhraseLower'] = df_test["Phrase"].str.lower()
    phrase_inter = set(df_train["PhraseLower"]).intersection(set(df_test["PhraseLower"]))
    print("Ratio of test set examples which occur in the train set: {0:.2f}".format(len(phrase_inter)/df_test.shape[0]))
    df_test = pd.merge(df_test, df_train[["PhraseLower", "Sentiment"]], on="PhraseLower", how="left")

    texts_test = np.array(list(vocab.transform(list(df_test.Phrase))))
    dence_test = transform_dence(df_test)
    
    texts, sid, label, dence = prepare_train(df_train)
    texts, label = np.array(list(vocab.transform(texts))), label_ed.transform(label)

    results = []
    for train_index, dev_index in kfold_by_sentence(sid, n_splits):
        texts_train, label_train, dence_train = texts[train_index], label[train_index], dence[train_index]
        texts_dev, label_dev, dence_dev = texts[dev_index], label[dev_index], dence[dev_index]
        
        hparams_cv = HParams(**hparams.values())
        feeder_train = TrainInputFeeder2X(texts_train, dence_train, label_train, 
                                            num_epochs=hparams_cv.num_epochs, batch_size=hparams_cv.batch_size, 
                                            shuffle=hparams_cv.shuffle)
        feeder_pred = PredictInputFeeder2X(texts_test, dence_test, batch_size=2048)
        
        results.append(train_predict_cv(hparams_cv, feeder_train, texts_dev, label_dev, dence_dev, feeder_pred))
    
    result = sum(results)
    labels_pred = label_ed.reverse(result)
    
    df_test['pred'] = labels_pred
    df_test.loc[df_test["Sentiment"].isnull(), "Sentiment"] = df_test.loc[df_test["Sentiment"].isnull(), "pred"]
    
    df_test['Sentiment'] = df_test.Sentiment.astype(int)
    df = df_test.loc[:, ('PhraseId', 'Sentiment')]

    df.to_csv(path_predict, ',', index=False)
    
    if os.path.exists(os.path.join(".", "runs")):
        shutil.rmtree(os.path.join(".", "runs"))
    
    print('Submissions: %s' % path_predict)

hparams = HParams(
        embed_type='fast-text', embedding_keep_prob=0.5,
        num_units=128, attention_size=128,
        filter_sizes=(2, 3), num_filters=128,
        num_outputs_fc1=256, num_outputs_fc2=128, num_outputs_fc3=50, fc_keep_prob=0.8,
        learning_rate=0.001, batch_size=1024, num_epochs=15, shuffle=True, evaluate_every=138, verbose=1,
        out_prefix='rnn_cnn', out_dir=None
    )
# df_train = load_train_df()
# df_test = load_test_df()

# df_train['PhraseLower'] = df_train["Phrase"].str.lower()
# df_test['PhraseLower'] = df_test["Phrase"].str.lower()
# phrase_inter = set(df_train["PhraseLower"]).intersection(set(df_test["PhraseLower"]))
# print("Ratio of test set examples which occur in the train set: {0:.2f}".format(len(phrase_inter)/df_test.shape[0]))
# df_test = pd.merge(df_test, df_train[["PhraseLower", "Sentiment"]], on="PhraseLower", how="left")

# texts_test = np.array(list(vocab.transform(list(df_test.Phrase))))
# dence_test = transform_dence(df_test)

# texts, sid, label, dence = prepare_train(df_train)
# texts, label = np.array(list(vocab.transform(texts))), label_ed.transform(label)
# hparams.verbose = 2
# for train_index, dev_index in kfold_by_sentence(sid, 10):
#     texts_train, label_train, dence_train = texts[train_index], label[train_index], dence[train_index]
#     texts_dev, label_dev, dence_dev = texts[dev_index], label[dev_index], dence[dev_index]
        
#     hparams_cv = HParams(**hparams.values())

#     feeder_train = TrainInputFeeder2X(texts_train, dence_train, label_train, 
#                                             num_epochs=hparams_cv.num_epochs, batch_size=hparams_cv.batch_size, 
#                                             shuffle=hparams_cv.shuffle)
#     feeder_pred = PredictInputFeeder2X(texts_test, dence_test, batch_size=2048)
        
#     train_predict_cv(hparams_cv, feeder_train, texts_dev, label_dev, dence_dev, feeder_pred)
# shutil.rmtree(os.path.join(".", "runs"))
blend_cv(hparams, 10, 'submission_20181004_1.csv')