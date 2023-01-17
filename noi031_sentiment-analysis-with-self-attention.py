import numpy as np

import pandas

import tensorflow as tf

import re

import keras

from keras.preprocessing.sequence import pad_sequences

from keras.engine.topology import Layer

from keras.layers import *

from keras import backend as K

from keras.models import Model

import gensim

from gensim.models import Word2Vec

import math
ct_map = {"ain't": "is not", "'cause": "because","he's": "he is", "how'd": "how did","how's": "how is",

          "'m": " am", "let's": "let us", "ma'am": "madam", "o'clock": "of the clock","shan't": "shall not",

          "so's": "so as", "this's": "this is", "that's": "that is", "there's": "there is",

          "here's": "here is", "what's": "what is", "when's": 'when is', "where'd": "where did",

          "where's": "where is", "who's": "who is", "why's": "why is", "y'all": "you all", "'d": ' would',

          "'ll": ' will', "n't": " not", "'ve": " have", "'re": ' are'}
def CleanSentence(string):

    for a, b in ct_map.items():

        string = string.replace(a, b)#Contraction Recovery

    string = string.replace('!', ' puncexc')

    string = string.replace('?', ' puncask')

    string = re.sub(r'[0-9]+', '0', string)

    string = string.replace(' ', '9')#'9' here is just a special symbol

    string = ''.join(list(filter(str.isalnum, string)))

    return string.split('9')
def TextToSequence(data, vocab):

    return [list(filter(lambda x: x is not None, [vocab[token].index + 1 if token in vocab.keys() else None 

                                                  for token in sentence]))for sentence in data]
def textpre(emb_dim):

    train = pandas.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv','\t')

    unlabel_train = pandas.read_csv('../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv','\t',

                                    error_bad_lines = False)

    additional_data = pandas.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

    test = pandas.read_csv('../input/word2vec-nlp-tutorial/testData.tsv','\t')

    s = train['sentiment']

    del train['sentiment']

    train['sentiment'] = s #to rearrange the postion of column 'sentiment'

    unlabel_train['sentiment'] = -2

    additional_data['sentiment'] = np.asarray(additional_data['sentiment'] == 'positive', dtype = 'int32')

    s = np.asarray(np.asarray(test['id'].apply(lambda x: x.split('_')[1]), dtype = 'int32') > 5,

                   dtype = 'int32')

    #To deal with data leakage.

    #See https://www.kaggle.com/c/word2vec-nlp-tutorial/discussion/27022#latest-400953

    test['sentiment'] = -1

    del train['id'], test['id'], unlabel_train['id']

    data = pandas.concat([train, additional_data, unlabel_train, test])

    review = list(data['review'].apply(lambda x: CleanSentence(x)))

    model = Word2Vec(review, size = emb_dim, workers = 4, iter = 0)

    emb_mat = np.concatenate([np.zeros((1, emb_dim), dtype = 'float32'), model.wv.vectors])

    print(emb_mat.shape)

    seq = TextToSequence(review, model.wv.vocab)

    seq = pad_sequences([doc if len(doc) <= 128 else doc[:64] + doc[-64:] for doc in seq], padding = 'post')

    tar = np.asarray(data['sentiment'], dtype = 'int32')

    r = data['sentiment'] >= 0

    train = np.hstack((seq[r], np.expand_dims(tar[r], -1)))

    r = data['sentiment'] == -1

    test = np.hstack((seq[r], np.expand_dims(s, -1)))

    print('text preprocessing completed')

    return [train, test, emb_mat]
class Self_Attention(Layer):

    def __init__(self, head_num, head_size, **kwargs):

        self.head_size = head_size

        self.sqrt_head_size = math.sqrt(head_size)

        self.head_num = head_num

        self.output_units = head_size * head_num

        self.supports_masking = True

        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.d_model = input_shape[-1]

        self.kernal = self.add_weight('kernal', (self.head_num, 3, self.d_model, self.head_size),

                                      initializer = 'glorot_uniform')

        super(Self_Attention, self).build(input_shape)

        self.built = True

    def call(self, inputs, mask = None):

        w = K.dot(inputs, self.kernal)

        w = K.permute_dimensions(w, [3, 2, 0, 1, 4])

        Head = K.batch_dot(w[0], K.permute_dimensions(w[1], [0, 1, 3, 2]))

        if mask is not None:

            new_mask = K.expand_dims(mask, -1)

            new_mask = K.batch_dot(new_mask, K.permute_dimensions(new_mask, [0, 2, 1]))

            new_mask = K.expand_dims(new_mask, 0)

            Head -= 1e8 * (K.ones_like(Head) - new_mask)

        Head = K.softmax(Head)

        Head = K.batch_dot(Head, w[2])

        Head = K.permute_dimensions(Head, [1, 2, 3, 0])

        return K.reshape(Head, (-1, K.shape(Head)[1],self.output_units))

    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[1], self.output_units)

    def compute_mask(self, inputs, mask = None):

        pass
class Position_Embedding(Layer):

    def __init__(self, **kwargs):

        self.supports_masking = True

        super(Position_Embedding, self).__init__(**kwargs)

    def build(self, input_shape):

        self.depth = input_shape[1]

        self.d_model = input_shape[2]

        super(Position_Embedding, self).build(input_shape)

        self.built = True

    def call(self, inputs):

        pos_emb = np.zeros([1, self.depth, self.d_model])

        p = np.zeros([self.d_model], dtype = 'int32')

        for i in range(self.d_model):

            p[i] = math.pow(10000, (i - 1) / self.d_model) if i%2 else math.pow(10000, i / self.d_model)

        for i in range(self.depth):

            for j in range(self.d_model):

                pos_emb[0][i][j] = math.cos(i / p[j]) if j%2 else math.sin(i / p[j])

        return inputs + tf.Variable(np.asarray(pos_emb, dtype = 'float32'), trainable = False)

    def compute_mask(self, inputs, mask = None):

        pass
def Attention_block(inputs, head_num, head_size, hidden_units):

    s = Self_Attention(head_num, head_size)(inputs)

    s = BatchNormalization()(Add()([s, inputs]))

    output = Dense(hidden_units, activation = 'relu')(s)

    output = Dense(int(inputs.shape[-1]))(output)

    return BatchNormalization()(Add()([s, output]))
train, test, emb_mat = textpre(256)

tf.reset_default_graph()

keras.backend.clear_session()

np.random.seed(7)

tf.set_random_seed(7)

x = Input(shape = (train.shape[1] - 1,), dtype = 'int32')

emb = Embedding(emb_mat.shape[0], emb_mat.shape[1], weights = [emb_mat], mask_zero = True,

                trainable = False)(x)

prob = Position_Embedding()(BatchNormalization()(emb))

prob = Attention_block(prob, 4, 64, 1024)

prob = GlobalMaxPool1D()(prob)

prob = Dense(1, activation = 'sigmoid')(prob)

model = Model(inputs = x, outputs = prob)

model.compile('adam', 'binary_crossentropy', ['accuracy'])

model.summary()

model.fit(train[:, :-1], train[:, -1], 128, 12, validation_data = (test[:, :-1], test[:, -1]))#epoch 12

#make submission

res = np.asarray(model.predict(test[:, :-1]) > 0.5, dtype = 'int32')

p = pandas.read_csv('../input/word2vec-nlp-tutorial/sampleSubmission.csv')

p['sentiment'] = res[:, 0]

p.to_csv('submission.csv', index = False)