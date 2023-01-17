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
import numpy as np
np.random.seed(1337)  # for reproducibility
def load_data(path, num_words=None, skip_top=0, seed=1337):
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]
    
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]
    
    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])
    
    if not num_words:
        num_words = max([max(x) for x in xs])

    xs = [[w for w in x if skip_top <= w < num_words] for x in xs]
    
    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
    
    return (x_train, y_train), (x_test, y_test)
from keras.preprocessing import sequence
def get_data(maxlen = 80, max_features = 20000):
    #print('Loading data...')
    (X_train, y_train), (X_test, y_test) = load_data('../input/keras-imdb-reviews/imdb.npz',
                                                     num_words=max_features)
    #print(len(X_train), 'train sequences')
    #print(len(X_test), 'test sequences')
    
    #print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    #print('X_train shape:', X_train.shape)
    #print('X_test shape:', X_test.shape)
    return (X_train, y_train), (X_test, y_test)
max_features = 20000
print('Loading word index...')
import json
with open('../input/keras-imdb-reviews/imdb_word_index.json') as f:
    word_index = json.load(f)
    
print('Loading embeddings...')
embeddings_index = dict()
f = open('../input/glove6b300dtxt/glove.6B.300d.txt')

for line in f:
    # Note: use split(' ') instead of split() if you get an error.
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix
embedding_matrix = np.zeros((max_features, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        if i < max_features:
            embedding_matrix[i] = embedding_vector
            
print('Done loading embeddings')
import numpy as np

from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec

from keras.utils.conv_utils import conv_output_length

#import theano
#import theano.tensor as T


def _dropout(x, level, noise_shape=None, seed=None):
    x = K.dropout(x, level, noise_shape, seed)
    x *= (1. - level) # compensate for the scaling by the dropout
    return x


class QRNN(Layer):
    '''Quasi RNN

    # Arguments
        units: dimension of the internal projections and the final output.

    # References
        - [Quasi-recurrent Neural Networks](http://arxiv.org/abs/1611.01576)
    '''
    def __init__(self, units, window_size=2, stride=1,
                 return_sequences=False, go_backwards=False, 
                 stateful=False, unroll=False, activation='tanh',
                 kernel_initializer='uniform', bias_initializer='zero',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,kernel_constraint=None, bias_constraint=None, 
                 dropout=0, use_bias=True, input_dim=None, input_length=None,**kwargs):
        # Setup
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.units = units 
        self.window_size = window_size
        self.strides = (stride, 1)

        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(QRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None #Stateful RNNs need to know BS
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        kernel_shape = (self.window_size, 1, self.input_dim, self.units * 3)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', 
                                        shape=(self.units * 3,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        length = input_shape[1]
        if length:
            length = conv_output_length(length + self.window_size - 1,
                                        self.window_size, 'valid',
                                        self.strides[0])
        if self.return_sequences:
            return (input_shape[0], length, self.units)
        else:
            return (input_shape[0], self.units)

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not self.input_spec:
            raise RuntimeError('Layer has never been called '
                               'and thus has no states.')

        batch_size = self.input_spec.shape[0]
        if not batch_size:
            raise ValueError('If a QRNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')

        if self.states[0] is None:
            self.states = [K.zeros((batch_size, self.units))
                           for _ in self.states]
        elif states is None:
            for state in self.states:
                K.set_value(state, np.zeros((batch_size, self.units)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 'state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if value.shape != (batch_size, self.units):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, self.units)) +
                                     ', found shape=' + str(value.shape))
                K.set_value(state, value)

    def __call__(self, inputs, initial_state=None, **kwargs):
        # If `initial_state` is specified,
        # and if it a Keras tensor,
        # then add it to the inputs and temporarily
        # modify the input spec to include the state.
        if initial_state is not None:
            if hasattr(initial_state, '_keras_history'):
                # Compute the full input spec, including state
                input_spec = self.input_spec
                state_spec = self.state_spec
                if not isinstance(state_spec, list):
                    state_spec = [state_spec]
                self.input_spec = [input_spec] + state_spec

                # Compute the full inputs, including state
                if not isinstance(initial_state, (list, tuple)):
                    initial_state = [initial_state]
                inputs = [inputs] + list(initial_state)

                # Perform the call
                output = super(QRNN, self).__call__(inputs, **kwargs)

                # Restore original input spec
                self.input_spec = input_spec
                return output
            else:
                kwargs['initial_state'] = initial_state
        return super(QRNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, initial_state=None, training=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_states = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(inputs)

        if len(initial_states) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_states)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        
        preprocessed_input = self.preprocess_input(inputs, training=None) #Convolutional Step

        last_output, outputs, states = K.rnn(self.step, preprocessed_input, #fo-gate step
                                            initial_states,
                                            go_backwards=self.go_backwards,
                                            mask=mask,
                                            constants=constants,
                                            unroll=self.unroll,
                                            input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def preprocess_input(self, inputs, training=None):
        if self.window_size > 1: # Pad
            inputs = K.temporal_padding(inputs, (self.window_size-1, 0))
        inputs = K.expand_dims(inputs, 2)  # add a dummy dimension

        output = K.conv2d(inputs, self.kernel, strides=self.strides, # Conv Step
                          padding='valid',
                          data_format='channels_last')
        output = K.squeeze(output, 2)  # remove the dummy dimension
        
        if self.use_bias: #Add bias
            output = K.bias_add(output, self.bias, data_format='channels_last')
        
        return output

    def step(self, inputs, states): #Fo-pool implementation
        prev_output = states[0]

        z = inputs[:, :self.units]
        f = inputs[:, self.units:2 * self.units]
        o = inputs[:, 2 * self.units:]

        z = self.activation(z)
        f = K.sigmoid(f) 
        o = K.sigmoid(o)

        output = f * prev_output + (1 - f) * z
        output = o * output

        return output, [output]

    def get_constants(self, inputs, training=None):
        return []
 
    def get_config(self):
        config = {'units': self.units,
                  'window_size': self.window_size,
                  'stride': self.strides[0],
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'use_bias': self.use_bias,
                  'activation': activations.serialize(self.activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(QRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Dropout
from keras.layers import CuDNNLSTM, LSTM
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.optimizers import RMSprop
def build_QRNN():
    #print('Build model QRNN')
    model = Sequential()
    model.add(Embedding(max_features, 300,weights=[embedding_matrix],trainable=False))
    model.add(Dropout(0.3))

    model.add(QRNN(256, window_size=3, dropout=0,return_sequences=True, 
                   kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                   kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dropout(0.3))

    model.add(QRNN(256, window_size=3, dropout=0,return_sequences=True, 
                   kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                   kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dropout(0.3))

    model.add(QRNN(256, window_size=3, dropout=0,return_sequences=True, 
                   kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                   kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dropout(0.3))

    model.add(QRNN(256, window_size=3, dropout=0,return_sequences=False, 
                   kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                   kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-8),
                  metrics=['accuracy'])
    
    return model
def build_LSTM():
    
    model = Sequential()
    model.add(Embedding(max_features, 300,weights=[embedding_matrix],trainable=False))
    model.add(Dropout(0.3))

    model.add(LSTM(256,return_sequences=True, 
                   kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                   kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dropout(0.3))

    model.add(LSTM(256,return_sequences=True, 
                   kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                   kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dropout(0.3))

    model.add(LSTM(256,return_sequences=True, 
                   kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                   kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dropout(0.3))

    model.add(LSTM(256 ,return_sequences=False, 
                   kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                   kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-8),
                  metrics=['accuracy'])
    
    return model
import time
from sklearn.utils import resample
def test_model(model, maxlen, batch_size,n_samples = 10000):
    print('Testing with Seq Len: {}, Batch Size: {}'.format(maxlen,batch_size))
    
    (X_train, y_train), (X_test, y_test) = get_data(maxlen=maxlen)
    X_train, y_train, X_test, y_test = resample(X_train, y_train, X_test, y_test,n_samples=n_samples,random_state = 1337)
    print('Dry Run')
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              epochs=1)
    
    print('On Time')
    start = time.time()
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              epochs=1)
    finish = time.time()
    total = finish-start
    print('Evaluating')
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    return total, score, acc
model = build_QRNN()
qrnn_512_8_total, score, qrnn_512_8_acc = test_model(model,maxlen=512,batch_size=8, n_samples=1000)

print('QRNN Run Time Per Epoch: {:.4f}, Accuracy after two epochs: {:.4f}'.format(qrnn_512_8_total,qrnn_512_8_acc))
model = build_QRNN()
qrnn_32_256_total, score, qrnn_32_256_acc = test_model(model,maxlen=32,batch_size=256, n_samples=1000)

print('QRNN Run Time Per Epoch: {:.4f}, Accuracy after two epochs: {:.4f}'.format(qrnn_32_256_total,qrnn_32_256_acc))
model = build_LSTM()
lstm_512_8_total, score, lstm_512_8_acc = test_model(model,maxlen=512,batch_size=8,n_samples=1000)

print('LSTM Run Time Per Epoch: {:.4f}, Accuracy after two epochs: {:.4f}'.format(lstm_512_8_total,lstm_512_8_acc))
model = build_LSTM()
lstm_32_256_total, score, lstm_32_256_acc = test_model(model,maxlen=32,batch_size=256,n_samples=1000)

print('LSTM Run Time Per Epoch: {:.4f}, Accuracy after two epochs: {:.4f}'.format(lstm_32_256_total,lstm_32_256_acc))
print('Speedup with Seq Len 512 and BS 8: {:.2f}'.format(lstm_512_8_total / qrnn_512_8_total))
print('Speedup with Seq Len 32 and BS 256: {:.2f}'.format(lstm_32_256_total / qrnn_32_256_total))
def build_CuDNNLSTM():
    
    model = Sequential()
    model.add(Embedding(max_features, 300,weights=[embedding_matrix],trainable=False))
    model.add(Dropout(0.3))

    model.add(CuDNNLSTM(256,return_sequences=True, 
                       kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                       kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dropout(0.3))

    model.add(CuDNNLSTM(256,return_sequences=True, 
                       kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                       kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dropout(0.3))

    model.add(CuDNNLSTM(256,return_sequences=True, 
                       kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                       kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dropout(0.3))

    model.add(CuDNNLSTM(256 ,return_sequences=False, 
                       kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6),
                       kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-8),
                  metrics=['accuracy'])
    
    return model
model = build_CuDNNLSTM()
culstm_512_8_total, score, culstm_512_8_acc = test_model(model,maxlen=512,batch_size=8,n_samples=1000)

print('CuDNNLSTM Run Time Per Epoch: {:.4f}, Accuracy after two epochs: {:.4f}'.format(culstm_512_8_total,culstm_512_8_acc))
model = build_CuDNNLSTM()
culstm_32_256_total, score, culstm_32_256_acc = test_model(model,maxlen=32,batch_size=256,n_samples=1000)

print('CuDNNLSTM Run Time Per Epoch: {:.4f}, Accuracy after two epochs: {:.4f}'.format(culstm_32_256_total,culstm_32_256_acc))
print('Speedup with Seq Len 512 and BS 8: {:.2f}'.format(culstm_512_8_total / qrnn_512_8_total))
print('Speedup with Seq Len 32 and BS 256: {:.2f}'.format(culstm_32_256_total / qrnn_32_256_total))
