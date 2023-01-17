import os
import numpy as np

# Keras imports

from keras.layers import Embedding, Dense, Input
from keras.layers import Bidirectional, TimeDistributed, CuDNNGRU, CuDNNLSTM, Convolution1D
from keras.layers import Conv1D, merge # old
from keras.layers import Flatten, concatenate, Dropout, PReLU, Activation, BatchNormalization
from keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D
from keras.engine import InputSpec, Layer
from keras.models import Model
from keras import initializers
from keras import constraints
from keras import optimizers
from keras import regularizers
from keras import backend as K

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# GRU-ATT

MAX_SENT_LENGTH = 50 
MAX_SENTS = 35

embedding_matrix = np.zeros((50000, 300)) # dummy

# https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py AND
# https://www.kaggle.com/hoonkeng/how-to-get-81-gru-att-lgbm-tf-idf-eda

class AttLayer(Layer):
  def __init__(self, use_bias=True, activation ='tanh', **kwargs):
    self.init = initializers.get('normal')
    self.use_bias = use_bias
    self.activation = activation
    super(AttLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    assert len(input_shape)==3
    self.W = self.add_weight(name='kernel', 
                             shape=(input_shape[-1],1),
                             initializer='normal',
                             trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(name='bias', 
                                  shape=(1,),
                                  initializer='zeros',
                                  trainable=True)
    else:
      self.bias = None
    super(AttLayer, self).build(input_shape) 

  def call(self, x, mask=None):
    eij = K.dot(x, self.W)
    if self.use_bias:
      eij = K.bias_add(eij, self.bias)
    if self.activation == 'tanh':
      eij = K.tanh(eij)
    elif self.activation =='relu':
      eij = K.relu(eij)
    else:
      eij = eij
    ai = K.exp(eij)
    weights = ai/K.sum(ai, axis=1, keepdims=True)
    weighted_input = x*weights
    return K.sum(weighted_input, axis=1)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[-1])

  def get_config(self):
    config = { 'activation': self.activation }
    base_config = super(AttLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
   
def get_attgru_model(gru_dense_dim = (64, 128), trainable=False, lr=0.0007, lr_decay=1e-16): 
  
  # Encoder
  
  sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32', name='main_input') 
  
  embedding_layer = Embedding(embedding_matrix.shape[0],
                              embedding_matrix.shape[1],
                              weights=[embedding_matrix],
                              input_length=MAX_SENT_LENGTH,
                              trainable=trainable)   

  embedded_sequences = embedding_layer(sentence_input)

  l_lstm = Bidirectional(CuDNNGRU(gru_dense_dim[0], return_sequences=True))(embedded_sequences)
  l_dense = TimeDistributed(Dense(gru_dense_dim[1]))(l_lstm)
  l_att = AttLayer()(l_dense)
  sentEncoder = Model(sentence_input, l_att)
  
  # Decoder
  
  review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
  review_encoder = TimeDistributed(sentEncoder)(review_input)
  l_lstm_sent = Bidirectional(CuDNNLSTM(gru_dense_dim[0], return_sequences=True))(review_encoder)
  l_dense_sent = TimeDistributed(Dense(gru_dense_dim[1]))(l_lstm_sent)
  l_att_sent = AttLayer()(l_dense_sent)
  preds = Dense(2, activation='softmax')(l_att_sent)
  sentDecoder = Model(review_input, preds)

  sentDecoder.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr, lr_decay),
                      metrics=['acc'])
  
  return sentEncoder, sentDecoder # fit on sentDecoder

sentEncoder, sentDecoder = get_attgru_model()

print('encoder:')
sentEncoder.summary()

print('decorder:')
sentDecoder.summary()
# Bi-LSTM

cat_features_hash = ['cat_%d' % n for n in range(1,11)] # dummy
max_cat_hash_size = 50000
num_features = ['num_%d' % n for n in range(1,401)] # dummy
MAX_WORDS = 500

embedding_matrix_2 = np.zeros((50000, 200)) # dummy

# https://www.kaggle.com/qinhui1999/deep-learning-is-all-you-need-lb-0-80x
# https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def get_model_bilstm(cat_embed_output_dim=10, trainable=False, gru_spec=(50, 100), gru_dropout=5e-5, lr=0.0006):
  input_cat = Input((len(cat_features_hash), ))
  input_num = Input((len(num_features), ))
  input_words = Input((MAX_WORDS, ))
  
  x_cat = Embedding(max_cat_hash_size, cat_embed_output_dim)(input_cat)    
  x_cat = SpatialDropout1D(0.3)(x_cat)
  x_cat = Flatten()(x_cat)
  
  x_words = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      trainable=trainable)(input_words)
  x_words = SpatialDropout1D(0.25)(x_words) # 0.30
  
  x_words = Bidirectional(CuDNNLSTM(gru_spec[0], return_sequences=True,
                          kernel_regularizer=regularizers.l2(gru_dropout),
                          recurrent_regularizer=regularizers.l2(gru_dropout)))(x_words)
  x_words = Convolution1D(gru_spec[1], 3, activation="relu")(x_words)
  
  x_words1_1 = GlobalMaxPool1D()(x_words)
  x_words1_2 = GlobalAveragePooling1D()(x_words)
  x_words1_3 = AttentionWithContext()(x_words)
  x_words = concatenate([x_words1_1, x_words1_2, x_words1_3])
  x_words = Dropout(0.25)(x_words)       
  x_words = Dense(100, activation="relu")(x_words) # 100
    
  if embedding_matrix_2 is not None:
    x_words_2 = Embedding(embedding_matrix_2.shape[0], embedding_matrix_2.shape[1],
                        weights=[embedding_matrix_2],
                        trainable=trainable)(input_words)
    x_words_2 = SpatialDropout1D(0.25)(x_words_2) # 0.30

    x_words_2 = Bidirectional(CuDNNLSTM(gru_spec[0], return_sequences=True,
                 kernel_regularizer=regularizers.l2(gru_dropout),
                 recurrent_regularizer=regularizers.l2(gru_dropout)))(x_words_2)
    x_words_2 = Convolution1D(gru_spec[1], 3, activation="relu")(x_words_2)
    
    x_words2_1 = GlobalMaxPool1D()(x_words_2)
    x_words2_2 = GlobalAveragePooling1D()(x_words_2)
    x_words2_3 = AttentionWithContext()(x_words_2)
    x_words_2 = concatenate([x_words2_1, x_words2_2, x_words2_3])
    x_words_2 = Dropout(0.25)(x_words_2)       
    x_words_2 = Dense(100, activation="relu")(x_words_2) # 100
   
    x_words = concatenate([x_words, x_words_2])

  # extra dense later to handle >400 numerical features
  x_num = Dense(200, activation="relu")(input_num)
  x_num = Dropout(0.25)(x_num)
  x_num = Dense(100, activation="relu")(x_num)

  x = concatenate([x_cat, x_num, x_words])

  x = Dense(50 + (0 if embedding_matrix_2 is None or gru_spec[1] == 0 else 14), activation="relu")(x) # was 30
  x = Dropout(0.25)(x)
  predictions = Dense(1, activation="sigmoid")(x)
  model = Model(inputs=[input_cat, input_num, input_words], outputs=predictions)
  model.compile(optimizer=optimizers.Adam(lr, decay=1e-6),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model

model = get_model_bilstm()
model.summary()
# Capsule
# https://www.kaggle.com/chongjiujjin/capsule-net-with-gru

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
      
def get_model_capsule(cat_embed_output_dim=10, trainable=False, gru_spec=(128, 0), gru_dropout=5e-5, lr=0.0007):
  input_cat = Input((len(cat_features_hash), ))
  input_num = Input((len(num_features), ))
  input_words = Input((MAX_WORDS, ))
  
  x_cat = Embedding(max_cat_hash_size, cat_embed_output_dim)(input_cat)    
  x_cat = SpatialDropout1D(0.3)(x_cat)
  x_cat = Flatten()(x_cat)
  
  x_words = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      trainable=trainable)(input_words)
  x_words = SpatialDropout1D(0.25)(x_words)
  
  # https://github.com/mattmotoki/toxic-comment-classification/blob/master/code/modeling/Refine CapsuleNet.ipynb
  x_words = Bidirectional(CuDNNGRU(gru_spec[0], return_sequences=True,
                          kernel_regularizer=regularizers.l2(gru_dropout),
                          recurrent_regularizer=regularizers.l2(gru_dropout)))(x_words)
  x_words = PReLU()(x_words)
  x_words = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(x_words)
  x_words = Flatten()(x_words)
  x_words = Dropout(0.15)(x_words)
     
  x_cat = Dense(100, activation="relu")(x_cat)
  
  # extra dense later to handle >400 numerical features
  x_num = Dense(200, activation="relu")(input_num)
  x_num = Dropout(0.25)(x_num)
  x_num = Dense(100, activation="relu")(x_num)

  x = concatenate([x_cat, x_num, x_words])

  x = Dense(50 + (0 if embedding_matrix_2 is None or gru_spec[1] == 0 else 14), activation="relu")(x) # was 30
  x = Dropout(0.25)(x)
  predictions = Dense(1, activation="sigmoid")(x)
  model = Model(inputs=[input_cat, input_num, input_words], outputs=predictions)
  model.compile(optimizer=optimizers.Adam(lr, decay=1e-6),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model

model = get_model_capsule()
model.summary()
# Combined Bi-GRU and Conv1D
# https://www.kaggle.com/fizzbuzz/the-all-in-one-model

def get_model_bigru_conv1d(cat_embed_output_dim=32, 
                           trainable=False,
                           recurrent_units = 96,
                           convolution_filters = 192,
                           dense_units = [256, 128, 64],
                           dropout_rate = 0.3,
                           lr = 0.0006):
  
  input_cat = Input((len(cat_features_hash), ))
  input_num = Input((len(num_features), ))
  input_words = Input((MAX_WORDS, ))
  
  x_cat = Embedding(max_cat_hash_size, cat_embed_output_dim)(input_cat)
  x_cat = SpatialDropout1D(dropout_rate)(x_cat)
  x_cat = Flatten()(x_cat)
  
  x_words = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], # max_features, maxlen,
                          weights=[embedding_matrix],
                          trainable=trainable)(input_words)
  x_words = SpatialDropout1D(dropout_rate)(x_words)
  
  x_words1 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x_words)
  x_words1 = Convolution1D(convolution_filters, 3, activation="relu")(x_words1)
  x_words1_1 = GlobalMaxPool1D()(x_words1)
  x_words1_2 = GlobalAveragePooling1D()(x_words1)
  x_words1_3 = AttentionWithContext()(x_words1)
  
  x_words2 = Convolution1D(convolution_filters, 2, activation="relu")(x_words)
  x_words2 = Convolution1D(convolution_filters, 2, activation="relu")(x_words2)
  x_words2_1 = GlobalMaxPool1D()(x_words2)
  x_words2_2 = GlobalAveragePooling1D()(x_words2)
  x_words2_3 = AttentionWithContext()(x_words2)
  
  x_num = input_num

  x = concatenate([x_words1_1, x_words1_2, x_words1_3, x_words2_1, x_words2_2, x_words2_3, x_cat, x_num])
  x = BatchNormalization()(x)
  x = Dense(dense_units[0], activation="relu")(x)
  x = Dense(dense_units[1], activation="relu")(x)
  
  x = concatenate([x, x_num])
  x = Dense(dense_units[2], activation="relu")(x)
  predictions = Dense(1, activation="sigmoid")(x)
  model = Model(inputs=[input_cat, input_num, input_words], outputs=predictions)
  model.compile(optimizer=optimizers.Adam(lr, decay=1e-6),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model  

model = get_model_bigru_conv1d()
model.summary()