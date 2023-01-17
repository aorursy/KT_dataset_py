!pip install --upgrade tensorflow
import tensorflow
tensorflow.__version__
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -*- coding: utf-8 -*-

"""
Author: Philipp Gross, https://github.com/fchollet/keras/pull/4621/files
"""

from __future__ import absolute_import
from __future__ import print_function

from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec


import tensorflow as tf

def logsumexp(x, axis=None):
    '''Returns `log(sum(exp(x), axis=axis))` with improved numerical stability.
    '''
    return tf.reduce_logsumexp(x, axis=[axis])


def batch_gather(reference, indices):
    '''Batchwise gathering of row indices.

    The numpy equivalent is reference[np.arange(batch_size), indices].

    # Arguments
        reference: tensor with ndim >= 2 of shape
            (batch_size, dim1, dim2, ..., dimN)
        indices: 1d integer tensor of shape (batch_size) satisfiying
            0 <= i < dim2 for each element i.

    # Returns
        A tensor with shape (batch_size, dim2, ..., dimN)
        equal to reference[1:batch_size, indices]
    '''
    batch_size = K.shape(reference)[0]
    indices = tf.stack([tf.range(batch_size), indices], axis=1)
    return tf.gather_nd(reference, indices)


def path_energy(y, x, U, b_start=None, b_end=None, mask=None):
    '''Calculates the energy of a tag path y for a given input x (with mask),
    transition energies U and boundary energies b_start, b_end.'''
    x = add_boundary_energy(x, b_start, b_end, mask)
    return path_energy0(y, x, U, mask)


def path_energy0(y, x, U, mask=None):
    '''Path energy without boundary potential handling.'''
    n_classes = K.shape(x)[2]
    y_one_hot = K.one_hot(y, n_classes)

    # Tag path energy
    energy = K.sum(x * y_one_hot, 2)
    energy = K.sum(energy, 1)

    # Transition energy
    y_t = y[:, :-1]
    y_tp1 = y[:, 1:]
    U_flat = K.reshape(U, [-1])
    # Convert 2-dim indices (y_t, y_tp1) of U to 1-dim indices of U_flat:
    flat_indices = y_t * n_classes + y_tp1
    U_y_t_tp1 = K.gather(U_flat, flat_indices)

    if mask is not None:
        mask = K.cast(mask, K.floatx())
        y_t_mask = mask[:, :-1]
        y_tp1_mask = mask[:, 1:]
        U_y_t_tp1 *= y_t_mask * y_tp1_mask

    energy += K.sum(U_y_t_tp1, axis=1)

    return energy


def sparse_chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    '''Given the true sparsely encoded tag sequence y, input x (with mask),
    transition energies U, boundary energies b_start and b_end, it computes
    the loss function of a Linear Chain Conditional Random Field:
    loss(y, x) = NNL(P(y|x)), where P(y|x) = exp(E(y, x)) / Z.
    So, loss(y, x) = - E(y, x) + log(Z)
    Here, E(y, x) is the tag path energy, and Z is the normalization constant.
    The values log(Z) is also called free energy.
    '''
    x = add_boundary_energy(x, b_start, b_end, mask)
    energy = path_energy0(y, x, U, mask)
    energy -= free_energy0(x, U, mask)
    return K.expand_dims(-energy, -1)


def chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    '''Variant of sparse_chain_crf_loss but with one-hot encoded tags y.'''
    y_sparse = K.argmax(y, -1)
    y_sparse = K.cast(y_sparse, 'int32')
    return sparse_chain_crf_loss(y_sparse, x, U, b_start, b_end, mask)


def add_boundary_energy(x, b_start=None, b_end=None, mask=None):
    '''Given the observations x, it adds the start boundary energy b_start (resp.
    end boundary energy b_end on the start (resp. end) elements and multiplies
    the mask.'''
    if mask is None:
        if b_start is not None:
            x = K.concatenate([x[:, :1, :] + b_start, x[:, 1:, :]], axis=1)
        if b_end is not None:
            x = K.concatenate([x[:, :-1, :], x[:, -1:, :] + b_end], axis=1)
    else:
        mask = K.cast(mask, K.floatx())
        mask = K.expand_dims(mask, 2)
        x *= mask
        if b_start is not None:
            mask_r = K.concatenate([K.zeros_like(mask[:, :1]), mask[:, :-1]], axis=1)
            start_mask = K.cast(K.greater(mask, mask_r), K.floatx())
            x = x + start_mask * b_start
        if b_end is not None:
            mask_l = K.concatenate([mask[:, 1:], K.zeros_like(mask[:, -1:])], axis=1)
            end_mask = K.cast(K.greater(mask, mask_l), K.floatx())
            x = x + end_mask * b_end
    return x


def viterbi_decode(x, U, b_start=None, b_end=None, mask=None):
    '''Computes the best tag sequence y for a given input x, i.e. the one that
    maximizes the value of path_energy.'''
    x = add_boundary_energy(x, b_start, b_end, mask)

    alpha_0 = x[:, 0, :]
    gamma_0 = K.zeros_like(alpha_0)
    initial_states = [gamma_0, alpha_0]
    _, gamma = _forward(x,
                        lambda B: [K.cast(K.argmax(B, axis=1), K.floatx()), K.max(B, axis=1)],
                        initial_states,
                        U,
                        mask)
    y = _backward(gamma, mask)
    return y


def free_energy(x, U, b_start=None, b_end=None, mask=None):
    '''Computes efficiently the sum of all path energies for input x, when
    runs over all possible tag sequences.'''
    x = add_boundary_energy(x, b_start, b_end, mask)
    return free_energy0(x, U, mask)


def free_energy0(x, U, mask=None):
    '''Free energy without boundary potential handling.'''
    initial_states = [x[:, 0, :]]
    last_alpha, _ = _forward(x,
                             lambda B: [logsumexp(B, axis=1)],
                             initial_states,
                             U,
                             mask)
    return last_alpha[:, 0]


def _forward(x, reduce_step, initial_states, U, mask=None):
    '''Forward recurrence of the linear chain crf.'''

    def _forward_step(energy_matrix_t, states):
        alpha_tm1 = states[-1]
        new_states = reduce_step(K.expand_dims(alpha_tm1, 2) + energy_matrix_t)
        return new_states[0], new_states

    U_shared = K.expand_dims(K.expand_dims(U, 0), 0)

    if mask is not None:
        mask = K.cast(mask, K.floatx())
        mask_U = K.expand_dims(K.expand_dims(mask[:, :-1] * mask[:, 1:], 2), 3)
        U_shared = U_shared * mask_U

    inputs = K.expand_dims(x[:, 1:, :], 2) + U_shared
    inputs = K.concatenate([inputs, K.zeros_like(inputs[:, -1:, :, :])], axis=1)

    last, values, _ = K.rnn(_forward_step, inputs, initial_states)
    return last, values


def _backward(gamma, mask):
    '''Backward recurrence of the linear chain crf.'''
    gamma = K.cast(gamma, 'int32')

    def _backward_step(gamma_t, states):
        y_tm1 = K.squeeze(states[0], 0)
        y_t = batch_gather(gamma_t, y_tm1)
        return y_t, [K.expand_dims(y_t, 0)]

    initial_states = [K.expand_dims(K.zeros_like(gamma[:, 0, 0]), 0)]
    _, y_rev, _ = K.rnn(_backward_step,
                        gamma,
                        initial_states,
                        go_backwards=True)
    y = K.reverse(y_rev, 1)

    if mask is not None:
        mask = K.cast(mask, dtype='int32')
        # mask output
        y *= mask
        # set masked values to -1
        y += -(1 - mask)
    return y


class ChainCRF(Layer):
    '''A Linear Chain Conditional Random Field output layer.
    It carries the loss function and its weights for computing
    the global tag sequence scores. While training it acts as
    the identity function that passes the inputs to the subsequently
    used loss function. While testing it applies Viterbi decoding
    and returns the best scoring tag sequence as one-hot encoded vectors.
    # Arguments
        init: weight initializers function for chain energies U.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializers](../initializers.md)).
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the transition weight matrix.
        b_start_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the start bias b.
        b_end_regularizer: instance of [WeightRegularizer](../regularizers.md)
            module, applied to the end bias b.
        b_start_constraint: instance of the [constraints](../constraints.md)
            module, applied to the start bias b.
        b_end_regularizer: instance of the [constraints](../constraints.md)
            module, applied to the end bias b.
        weights: list of Numpy arrays for initializing [U, b_start, b_end].
            Thus it should be a list of 3 elements of shape
            [(n_classes, n_classes), (n_classes, ), (n_classes, )]
    # Input shape
        3D tensor with shape `(nb_samples, timesteps, nb_classes)`, where
        ´timesteps >= 2`and `nb_classes >= 2`.
    # Output shape
        Same shape as input.
    # Masking
        This layer supports masking for input sequences of variable length.
    # Example
    ```python
    # As the last layer of sequential layer with
    # model.output_shape == (None, timesteps, nb_classes)
    crf = ChainCRF()
    model.add(crf)
    # now: model.output_shape == (None, timesteps, nb_classes)
    # Compile model with chain crf loss (and one-hot encoded labels) and accuracy
    model.compile(loss=crf.loss, optimizer='sgd', metrics=['accuracy'])
    # Alternatively, compile model with sparsely encoded labels and sparse accuracy:
    model.compile(loss=crf.sparse_loss, optimizer='sgd', metrics=['sparse_categorical_accuracy'])
    ```
    # Gotchas
    ## Model loading
    When you want to load a saved model that has a crf output, then loading
    the model with 'keras.models.load_model' won't work properly because
    the reference of the loss function to the transition parameters is lost. To
    fix this, you need to use the parameter 'custom_objects' as follows:
    ```python
    from keras.layer.crf import create_custom_objects:
    model = keras.models.load_model(filename, custom_objects=create_custom_objects())
    ```
    ## Temporal sample weights
    Given a ChainCRF instance crf both loss functions, crf.loss and crf.sparse_loss
    return a tensor of shape (batch_size, 1) and not (batch_size, maxlen).
    that sample weighting in temporal mode.
    '''
    def __init__(self, init='glorot_uniform',
                 U_regularizer=None, b_start_regularizer=None, b_end_regularizer=None,
                 U_constraint=None, b_start_constraint=None, b_end_constraint=None,
                 weights=None,
                 **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        self.input_spec = [InputSpec(ndim=3)]
        self.init = initializers.get(init)

        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_start_regularizer = regularizers.get(b_start_regularizer)
        self.b_end_regularizer = regularizers.get(b_end_regularizer)
        self.U_constraint = constraints.get(U_constraint)
        self.b_start_constraint = constraints.get(b_start_constraint)
        self.b_end_constraint = constraints.get(b_end_constraint)

        self.initial_weights = weights

        super(ChainCRF, self).__init__(**kwargs)

    def compute_mask(self, input, mask=None):
        if mask is not None:
            return K.any(mask, axis=1)
        return mask

    def _fetch_mask(self):
        assert self._inbound_nodes, 'CRF has not connected to any layer.'
        mask = None
        # if self._inbound_nodes:
        #     mask = self. get_input_mask_at(0)
        return mask

    def build(self, input_shape):
        assert len(input_shape) == 3
        n_classes = input_shape[2]
        n_steps = input_shape[1]
        assert n_classes >= 2
        assert n_steps is None or n_steps >= 2
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, n_steps, n_classes))]

        self.U = self.add_weight(shape=(n_classes, n_classes),
                                 initializer=self.init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer,
                                 constraint=self.U_constraint)

        self.b_start = self.add_weight(shape=(n_classes, ),
                                       initializer='zero',
                                       name='{}_b_start'.format(self.name),
                                       regularizer=self.b_start_regularizer,
                                       constraint=self.b_start_constraint)

        self.b_end = self.add_weight(shape=(n_classes, ),
                                     initializer='zero',
                                     name='{}_b_end'.format(self.name),
                                     regularizer=self.b_end_regularizer,
                                     constraint=self.b_end_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):
        y_pred = viterbi_decode(x, self.U, self.b_start, self.b_end, mask)
        nb_classes = self.input_spec[0].shape[2]
        y_pred_one_hot = K.one_hot(y_pred, nb_classes)
        return K.in_train_phase(x, y_pred_one_hot)

    def loss(self, y_true, y_pred):
        '''Linear Chain Conditional Random Field loss function.∂ç
        '''
        mask = self._fetch_mask()
        return chain_crf_loss(y_true, y_pred, self.U, self.b_start, self.b_end, mask)

    def sparse_loss(self, y_true, y_pred):
        '''Linear Chain Conditional Random Field loss function with sparse
        tag sequences.
        '''
        y_true = K.cast(y_true, 'int32')
        y_true = K.squeeze(y_true, 2)
        mask = self._fetch_mask()
        return sparse_chain_crf_loss(y_true, y_pred, self.U, self.b_start, self.b_end, mask)

    def get_config(self):
        config = {#'init': self.init.__name__,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_start_regularizer': self.b_start_regularizer.get_config() if self.b_start_regularizer else None,
                  'b_end_regularizer': self.b_end_regularizer.get_config() if self.b_end_regularizer else None,
                  'U_constraint': self.U_constraint.get_config() if self.U_constraint else None,
                  'b_start_constraint': self.b_start_constraint.get_config() if self.b_start_constraint else None,
                  'b_end_constraint': self.b_end_constraint.get_config() if self.b_end_constraint else None,
                  }
        base_config = super(ChainCRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_custom_objects():
    '''Returns the custom objects, needed for loading a persisted model.'''
    instanceHolder = {'instance': None}

    class ClassWrapper(ChainCRF):
        def __init__(self, *args, **kwargs):
            instanceHolder['instance'] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder['instance'], 'loss')
        return method(*args)

    def sparse_loss(*args):
        method = getattr(instanceHolder['instance'], 'sparse_loss')
        return method(*args)

    return {'ChainCRF': ClassWrapper, 'loss': loss, 'sparse_loss': sparse_loss}

if __name__ == '__main__':
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding
    import numpy as np
    vocab_size = 20
    n_classes = 11
    model = Sequential()
    model.add(Embedding(vocab_size, n_classes))
    layer = ChainCRF()
    model.add(layer)
    model.compile(loss=layer.loss, optimizer='sgd')

    # Train first mini batch
    batch_size, maxlen = 2, 2
    x = np.random.randint(1, vocab_size, size=(batch_size, maxlen))
    y = np.random.randint(n_classes, size=(batch_size, maxlen))
    y = np.eye(n_classes)[y]
    model.train_on_batch(x, y)
    
    
    
    print(x)
    print(y)
    

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this script uses pretrained model to segment Arabic dialect data.
# it takes the pretrained model trained on joint dialects and the 
# training vocab and produces segmented text
#
# Copyright (C) 2017, Qatar Computing Research Institute, HBKU, Qatar
# Las Update: Sun Oct 29 15:34:43 +03 2017
#
# BibTex: @inproceedings{samih2017learning,
#  title={Learning from Relatives: Unified Dialectal Arabic Segmentation},
#  author={Samih, Younes and Eldesouki, Mohamed and Attia, Mohammed and Darwish, Kareem and Abdelali, Ahmed and Mubarak, Hamdy and Kallmeyer, Laura},
#  booktitle={Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017)},
#  pages={432--441},
#  year={2017}}
#
from __future__ import print_function

__author__ = 'Ahmed Abdelali (aabdelali@hbku.edu.qa)'

import numpy as np
import sys
import os
import re
import argparse
from collections import Counter
from itertools import chain
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import codecs
import json
from time import gmtime, strftime
import datetime
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(1337)  # for reproducibility
#sys.stdin = codecs.getreader('utf-8')(sys.stdin)

def printf(format, *args):
    sys.stdout.write(format % args)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def valid_date(datestring):
    try:
        mat=re.match('^(\d{2})[/.-](\d{2})[/.-](\d{4})$', datestring)
        if mat is not None:
            datetime.datetime(*(map(int, mat.groups()[-1::-1])))
            return True
    except ValueError:
        pass
    return False

def valid_number(numstring):
    try:
        mat=re.match("^[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?$", numstring)
        if mat is not None:
            return True
    except ValueError:
        pass
    return False
    
def valide_time(timestring):
    try:
        mat=re.match('^(2[0-3]|[01]?[0-9]):([0-5]?[0-9])$', timestring)
        if mat is not None:
            datetime.time(*(map(int, mat.groups()[::])))
            return True
        mat=re.match('^(2[0-3]|[01]?[0-9]):([0-5]?[0-9]):([0-5]?[0-9])$', timestring)
        if mat is not None:
            datetime.time(*(map(int, mat.groups()[::])))
            return True
        mat=re.match('^(2[0-3]|[01]?[0-9]):([0-5]?[0-9]):([0-5]?[0-9]).([0-9]?[0-9])$', timestring)
        if mat is not None:
            datetime.time(*(map(int, mat.groups()[::])))
            return True
    except ValueError:
        pass        
    return False

def valid_email(emailstring):
    try:
        mat=re.match('^[^@]+@[^@]+\.[^@]+$',emailstring)
        if mat is not None:
            return True
    except ValueError:
        pass
    return False

def removeDiacritics(instring):
    return re.sub(r'[ـًٌٍَُِّْ]', '', instring)

def isDelimiter(ch):
    if(ord(ch) == 32): #\u0020
        return True
    elif(ord(ch)>=0 and ord(ch)<=47): #\u0000-\u002F
        return True
    elif(ord(ch)>=58 and ord(ch)<=64): #\u003A-\u0040
        return True
    elif(ord(ch)>=123 and ord(ch)<=187): #\u007B-\u00BB
        return True
    elif(ord(ch)>=91 and ord(ch)<=96): #\u005B-\u005D
        return True
    elif(ord(ch)>=1536 and ord(ch)<=1548): #\u0600-\u060C
        return True
    elif(ord(ch)>=1748 and ord(ch)<=1773): #\u06D4-\u06ED
        return True
    elif(ord(ch)==65279): #\ufeff
        return True
    else:
        return False

def tokenizeline(txtstring):
    elements =[]
    #Remove Kashida and diacritics.
    txtstring = removeDiacritics(txtstring)

    #Split on Arabic delimiters
    for aword in re.split(r'،|٫|٫|٬|؛',txtstring):
        for word in aword.split():
            #print("==>",word)
            if (word.startswith("#")
                or word.startswith("@")
                or word.startswith(":")
                or word.startswith(";")
                or word.startswith("http://")
                or word.startswith("https://")
                or valid_email(word)
                or valid_date(word)
                or valid_number(word)
                or valide_time(word)):

                elements.append(word)
            else:
                for elt in word_tokenize(word):
                    elements.append(elt)
    output = ''
    for elt in elements:
        output = output + ' ' + elt 
    return output


def getLabels(filepathT,filepathD):
    labels = []
    for line in open(filepathT):
        line = line.strip()
        if len(line) == 0:
            continue
        splits = line.split('\t')
        labels.append(splits[1].strip())
    for line in open(filepathD):
        line = line.strip()
        if len(line) == 0:
            continue
        splits = line.split('\t')
        labels.append(splits[1].strip())
    return list(set(labels))


def load_data(options,seg_tags):
    X_words_train, y_train = load_file(options.train)
    X_words_dev, y_dev = load_file(options.dev)

    index2word = _fit_term_index(X_words_train+X_words_dev, reserved=['<PAD>', '<UNK>'])
    word2index = _invert_index(index2word)

    index2pos = seg_tags
    pos2index = _invert_index(index2pos)

    X_words_train = np.array([[word2index[w] for w in words] for words in X_words_train])
    y_train = np.array([[pos2index[t] for t in s_tags] for s_tags in y_train])

    X_words_dev = np.array([[word2index[w] for w in words] for words in X_words_dev])
    y_dev = np.array([[pos2index[t] for t in s_tags] for s_tags in y_dev])

    return (X_words_train, y_train), (X_words_dev, y_dev),(index2word, index2pos),word2index


def _fit_term_index(terms, reserved=[], preprocess=lambda x: x):
    all_terms = chain(*terms)
    all_terms = map(preprocess, all_terms)
    term_freqs = Counter(all_terms).most_common()
    id2term = reserved + [term for term, tf in term_freqs]
    return id2term


def _invert_index(id2term):
    return {term: i for i, term in enumerate(id2term)}

def build_words(src,ref,pred):
    words = []
    rtags = []
    ptags = []
    w = ''
    r = ''
    p = ''
    for i in range(len(src)):
        if(src[i] == 'WB'):
            words.append(w)
            rtags.append(r)
            ptags.append(p)
            w = ''
            r = ''
            p = ''
        else:
            w += src[i]
            r += ref[i]
            p += pred[i]           

    return words, rtags, ptags

def load_file(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in open(path):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    words, tags = zip(*[zip(*row) for row in sentences])
    return words, tags

def load_lookuplist(path):
    """
    Load lookp list.
    """
    listwords = {}
    for line in open(path):
        line = line.rstrip()
        listwords[line.replace('+','')] = line     
    return listwords


def build_model(model_path, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim):

    model = Sequential()
    model.add(Embedding(max_features, word_embedding_dim, input_length=maxlen, name='word_emb', mask_zero=True))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(lstm_dim,return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(nb_seg_tags)))
    crf = ChainCRF()
    model.add(crf)
    model.compile(loss=crf.sparse_loss,
                   optimizer= RMSprop(0.01),
                   metrics=['sparse_categorical_accuracy'])
    #model.compile('adam', loss=crf.sparse_loss, metrics=['sparse_categorical_accuracy'])
    #early_stopping = EarlyStopping(patience=10, verbose=1)
    #checkpointer = ModelCheckpoint(options.model + "/seg_keras_weights.hdf5",verbose=1,save_best_only=True)
    #eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Loading saved model:'+model_path + '/seg_keras_weights.h5')

    #model.load_weights(model_path + '/seg_keras_weights.hdf5')

    return model
main_path = "/kaggle/input/"
class Args:
    train = main_path + "data/joint.trian.3"
    model = "/kaggle/output/models"
    epochs = 100
    task = "train"  #chnage the task name for evaluate/decode/train
    lookup = main_path+ "data/lookup_list.txt"
    dev = main_path+ "data/joint.dev.3"
    test = main_path+ "data/joint.dev.3"

def main():

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-t", "--train",  default=main_path+ "data/joint.trian.3", help="Train set location")
#     parser.add_argument("-d", "--dev",  default=main_path+ "data/joint.dev.3", help="Dev set location")
#     parser.add_argument("-s", "--test",   default=main_path+ "data/joint.trian.3", help="Test set location")
#     parser.add_argument("-m", "--model",  default=main_path+"models", help="Model location")
#     parser.add_argument("-l", "--lookup",  default=main_path+ "data/lookup_list.txt", help="Lookup list location")
#     parser.add_argument("-p", "--epochs",  default=100, type=int, help="Lookup list location")
#     parser.add_argument("-i", "--input",  default="", help="Input stdin or file")
#     parser.add_argument("-o", "--output", default="", help="output file")
#     parser.add_argument("-k","--task", choices=['train','evaluate','decode'], help="Choice for task (train, evaluate, decode)")

    options = Args()

    seg_tags = ['E', 'S', 'B', 'M', 'WB'] #['WB', 'S', 'B', 'E', 'M']
    idx2Label = {0:'E', 1:'S', 2:'B', 3:'M', 4:'WB'}
    label2Idx = {'E':0, 'S':1, 'B':2, 'M':3, 'WB':4}

    word_embedding_dim = 200
    lstm_dim = 200

    #print('Loading data...')
    (X_words_train,y_train), (X_words_dev,y_dev), (index2word, index2pos),word2index = load_data(options,seg_tags)

    seq_len = []
    for i in range(len(X_words_train)):
        seq_len.append(len(X_words_train[i]))
    #print("MaxLen Train:",max(seq_len))

    maxlen = max(seq_len)  # cut texts after this number of words (among top max_features most common words)
    maxlen = 500 # Set to 500 max num of chars in one line.

    result = map(len,X_words_train)

    max_features = len(index2word)
    nb_seg_tags = len(index2pos)

    eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Loading Lookup list....')
    lookupList = load_lookuplist(options.lookup)

    
    if(options.task == 'train'):
        X_words_train = sequence.pad_sequences(X_words_train, maxlen=maxlen, padding='post')
        y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post')
        y_train = np.expand_dims(y_train, -1)

        X_words_dev = sequence.pad_sequences(X_words_dev, maxlen=maxlen, padding='post')
        y_dev = sequence.pad_sequences(y_dev, maxlen=maxlen, padding='post')
        y_dev = np.expand_dims(y_dev, -1)

        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Build model...')
        model = build_model(options.model, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim)

        early_stopping = EarlyStopping(patience=10, verbose=1)
        checkpointer = ModelCheckpoint(options.model + '/'+options.train.split('/')[-1]+'_keras_weights.h5',verbose=1,save_best_only=True)

        model_json = model.to_json()
        with open('sg_keras_weights.json', 'w') as json_file:
            json_file.write(model_json)
        print("saved json")
        model.fit(x=X_words_train, y=y_train,
          validation_data=(X_words_dev, y_dev),
          verbose=1,
          batch_size=64,
          epochs=options.epochs,
          callbacks=[early_stopping, checkpointer])  
        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Save the trained model...')
    elif(options.task == 'decode'):
        model = build_model(options.model, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim)
        # json_file = open(options.model + '/'+options.train.split('/')[-1]+'_keras_weights.json', encoding ='utf8', mode='r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        
        # model = model_from_json(loaded_model_json)
        model.load_weights('sg_keras_weights.h5')

        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Loading input...')
        sentences = []
        sentences_len = []
        l  = 0
        for line in sys.stdin:
            sentence = []

            if len(line) < 2:
                print("")
                continue
            words = tokenizeline(line).strip().split()
            for word in words:
                    for ch in word:
                        sentence.append([ch,'WB'])
                        l = l + 1
                    sentence.append(['WB','WB'])
                    l = l + 1
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                    sentences_len.append(l)

        
        listwords,tags = zip(*[zip(*row) for row in sentences])
        
        X_words_test = np.array([[word2index.get(w, word2index['<UNK>']) for w in words] for words in listwords])
        X_words_test = sequence.pad_sequences(X_words_test, maxlen=maxlen, padding='post')

        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Decoding ...')
        test_y_pred = model.predict(X_words_test, batch_size=200).argmax(-1)[X_words_test > 0]

        in_data = []
        for i in range(len(X_words_test)):
            for j in range(len(X_words_test[i])):
                if X_words_test[i][j] > 0:
                    in_data.append(index2word[X_words_test[i][j]])

        listchars = []
        for words in listwords:
            for w in words:
                listchars.append(w)

        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Writing output ...')

        word = ''
        segt = ''
        sent = 0
        for i in range(len(test_y_pred)):
            if(idx2Label[test_y_pred[i]] in ('B','M')):
                segt += in_data[i]
                word += listchars[i]
            elif(idx2Label[test_y_pred[i]] in ('E','S') and idx2Label.get(test_y_pred[i+1]) !='WB'):
                segt += in_data[i]+'+'
                word += listchars[i]
            elif(idx2Label[test_y_pred[i]] in ('E','S') and idx2Label.get(test_y_pred[i+1]) =='WB'):
                segt += in_data[i]
                word += listchars[i]
            elif(idx2Label[test_y_pred[i]] == 'WB'):
                if(word in lookupList):
                    
                    printf('%s ',lookupList[word])
                else:
                    if('<UNK>' in segt):
                        segt = word
                    
                    printf('%s ',segt)  
                word = ''
                segt = ''            
            if(sentences_len[sent] == i):
                print('')
                sent = sent + 1 
        print('')

    elif(options.task == 'evaluate'):
        model = build_model(options.model, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim)
        # json_file = open(options.model + '/'+options.train.split('/')[-1]+'_keras_weights.json', encoding ='utf8', mode='r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        
        # model = model_from_json(loaded_model_json)
        model.load_weights(options.model + '/'+options.train.split('/')[-1]+'_keras_weights.h5')

        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Loading input...')
        X_words, y_test_ref = load_file(options.test)
        X_words_test = np.array([[(word2index[w] if w  in word2index else word2index['<UNK>']) for w in words] for words in X_words])
        X_words_test = sequence.pad_sequences(X_words_test, maxlen=maxlen, padding='post')
        #y_test = sequence.pad_sequences(y_test_ref, maxlen=maxlen, padding='post')
        #y_test = np.expand_dims(y_test, -1)


        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Decoding ...')
        y_test_pred = model.predict(X_words_test, batch_size=200).argmax(-1)[X_words_test > 0]

        preds = [idx2Label[x] for x in y_test_pred]
        srcs  = list(chain.from_iterable(words for words in X_words))
        refs  = list(chain.from_iterable(tags for tags in y_test_ref))

        #refs  = [(tag for tag in tags) for tags in y_test_ref]
        print("Evaluation Seg Charachters:")
        print(classification_report(
                refs,
                preds,
                digits=5,
                target_names=seg_tags
            ).split('\n')[-2:])
        print("Charachters Confusion Matrix:")
        print(confusion_matrix(
                refs,
                preds,
                labels=seg_tags
            ))
        outfile = open(options.test+'.out','w')
        for i in range(len(preds)):
            outfile.write('%s\t%s\t%s\n'%(srcs[i],refs[i],preds[i]))
        outfile.close()

        (words,rtags,ptags) = build_words(srcs,refs,preds)

        print("Evaluation Seg Words:")
        print(classification_report(
                rtags,
                ptags,
                digits=5,
                target_names=list(set(rtags+ptags))
            ).split('\n')[-2:])

        outfile = open(options.test+'.words.out','w')
        for i in range(len(words)):
            outfile.write('%s\t%s\t%s\n'%(words[i],rtags[i],ptags[i]))
        outfile.close()

if __name__ == "__main__":
    main()

