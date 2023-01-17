import math

import numpy as np



import keras

import keras.backend as K

from keras.layers import Dropout

from keras.layers import Layer, Conv1D, Dropout, Add, Input

from keras.initializers import Ones, Zeros



import tensorflow as tf
py_any = K.py_any

ndim = K.ndim
def batch_dot(x, y, axes=None):

    """Batchwise dot product.



    `batch_dot` is used to compute dot product of `x` and `y` when

    `x` and `y` are data in batch, i.e. in a shape of

    `(batch_size, :)`.

    `batch_dot` results in a tensor or variable with less dimensions

    than the input. If the number of dimensions is reduced to 1,

    we use `expand_dims` to make sure that ndim is at least 2.



    # Arguments

        x: Keras tensor or variable with `ndim >= 2`.

        y: Keras tensor or variable with `ndim >= 2`.

        axes: list of (or single) int with target dimensions.

            The lengths of `axes[0]` and `axes[1]` should be the same.



    # Returns

        A tensor with shape equal to the concatenation of `x`'s shape

        (less the dimension that was summed over) and `y`'s shape

        (less the batch dimension and the dimension that was summed over).

        If the final rank is 1, we reshape it to `(batch_size, 1)`.



    # Examples

        Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`

        `batch_dot(x, y, axes=1) = [[17], [53]]` which is the main diagonal

        of `x.dot(y.T)`, although we never have to calculate the off-diagonal

        elements.



        Shape inference:

        Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.

        If `axes` is (1, 2), to find the output shape of resultant tensor,

            loop through each dimension in `x`'s shape and `y`'s shape:



        * `x.shape[0]` : 100 : append to output shape

        * `x.shape[1]` : 20 : do not append to output shape,

            dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)

        * `y.shape[0]` : 100 : do not append to output shape,

            always ignore first dimension of `y`

        * `y.shape[1]` : 30 : append to output shape

        * `y.shape[2]` : 20 : do not append to output shape,

            dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)

        `output_shape` = `(100, 30)`



    ```python

        >>> x_batch = K.ones(shape=(32, 20, 1))

        >>> y_batch = K.ones(shape=(32, 30, 20))

        >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])

        >>> K.int_shape(xy_batch_dot)

        (32, 1, 30)

    ```

    """

    if isinstance(axes, int):

        axes = (axes, axes)

    x_ndim = ndim(x)

    y_ndim = ndim(y)

    if axes is None:

        # behaves like tf.batch_matmul as default

        axes = [x_ndim - 1, y_ndim - 2]

    if py_any([isinstance(a, (list, tuple)) for a in axes]):

        raise ValueError('Multiple target dimensions are not supported. ' +

                         'Expected: None, int, (int, int), ' +

                         'Provided: ' + str(axes))

    if x_ndim > y_ndim:

        diff = x_ndim - y_ndim

        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))

    elif y_ndim > x_ndim:

        diff = y_ndim - x_ndim

        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))

    else:

        diff = 0

    if ndim(x) == 2 and ndim(y) == 2:

        if axes[0] == axes[1]:

            out = tf.reduce_sum(tf.multiply(x, y), axes[0])

        else:

            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])

    else:

        if axes is not None:

            adj_x = None if axes[0] == ndim(x) - 1 else True

            adj_y = True if axes[1] == ndim(y) - 1 else None

        else:

            adj_x = None

            adj_y = None

        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)

    if diff:

        if x_ndim > y_ndim:

            idx = x_ndim + y_ndim - 3

        else:

            idx = x_ndim - 1

        out = tf.squeeze(out, list(range(idx, idx + diff)))

    if ndim(out) == 1:

        out = expand_dims(out, 1)

    return out
def shape_list(x):

    if K.backend() != 'theano':

        tmp = K.int_shape(x)

    else:

        tmp = x.shape

    tmp = list(tmp)

    tmp[0] = -1

    return tmp





def split_heads(x, n: int, k: bool = False):  # B, L, C

    x_shape = shape_list(x)

    m = x_shape[-1]

    new_x_shape = x_shape[:-1] + [n, m // n]

    new_x = K.reshape(x, new_x_shape)

    return K.permute_dimensions(new_x, [0, 2, 3, 1] if k else [0, 2, 1, 3])





def merge_heads(x):

    new_x = K.permute_dimensions(x, [0, 2, 1, 3])

    x_shape = shape_list(new_x)

    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]

    return K.reshape(new_x, new_x_shape)





# q and v are B, H, L, C//H ; k is B, H, C//H, L ; mask is B, 1, L, L

def scaled_dot_product_attention_tf(q, k, v, attn_mask, attention_dropout: float, neg_inf: float):

    w = batch_dot(q, k)  # w is B, H, L, L

    w = w / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))

    if attn_mask is not None:

        w = attn_mask * w + (1.0 - attn_mask) * neg_inf

    w = K.softmax(w)

    w = Dropout(attention_dropout)(w)

    return batch_dot(w, v)  # it is B, H, L, C//H [like v]





def scaled_dot_product_attention_th(q, k, v, attn_mask, attention_dropout: float, neg_inf: float):

    w = theano_matmul(q, k)

    w = w / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))

    if attn_mask is not None:

        attn_mask = K.repeat_elements(attn_mask, shape_list(v)[1], 1)

        w = attn_mask * w + (1.0 - attn_mask) * neg_inf

    w = K.T.exp(w - w.max()) / K.T.exp(w - w.max()).sum(axis=-1, keepdims=True)

    w = Dropout(attention_dropout)(w)

    return theano_matmul(w, v)





def multihead_attention(x, attn_mask, n_head: int, n_state: int, attention_dropout: float, neg_inf: float):

    _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]

    q = split_heads(_q, n_head)  # B, H, L, C//H

    k = split_heads(_k, n_head, k=True)  # B, H, C//H, L

    v = split_heads(_v, n_head)  # B, H, L, C//H

    if K.backend() == 'tensorflow':

        a = scaled_dot_product_attention_tf(q, k, v, attn_mask, attention_dropout, neg_inf)

    else:

        a = scaled_dot_product_attention_th(q, k, v, attn_mask, attention_dropout, neg_inf)

    return merge_heads(a)





def gelu(x):

    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))





# https://stackoverflow.com/a/42194662/2796084

def theano_matmul(a, b, _left=False):

    assert a.ndim == b.ndim

    ndim = a.ndim

    assert ndim >= 2

    if _left:

        b, a = a, b

    if ndim == 2:

        return K.T.dot(a, b)

    else:

        # If a is broadcastable but b is not.

        if a.broadcastable[0] and not b.broadcastable[0]:

            # Scan b, but hold a steady.

            # Because b will be passed in as a, we need to left multiply to maintain

            #  matrix orientation.

            output, _ = K.theano.scan(theano_matmul, sequences=[b], non_sequences=[a[0], 1])

        # If b is broadcastable but a is not.

        elif b.broadcastable[0] and not a.broadcastable[0]:

            # Scan a, but hold b steady.

            output, _ = K.theano.scan(theano_matmul, sequences=[a], non_sequences=[b[0]])

        # If neither dimension is broadcastable or they both are.

        else:

            # Scan through the sequences, assuming the shape for this dimension is equal.

            output, _ = K.theano.scan(theano_matmul, sequences=[a, b])

        return output
def _get_pos_encoding_matrix(max_len: int, d_emb: int) -> np.array:

    pos_enc = np.array(

        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in

         range(max_len)], dtype=np.float32)

    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i

    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1

    return pos_enc





class BertEmbedding(keras.layers.Layer):

    def __init__(self, output_dim: int = 768, dropout: float = 0.1, vocab_size: int = 30000,

                 max_len: int = 512, trainable_pos_embedding: bool = True, use_one_dropout: bool = False,

                 use_embedding_layer_norm: bool = False, layer_norm_epsilon: float = 1e-5, **kwargs):

        super().__init__(**kwargs)

        self.max_len = max_len

        self.use_one_dropout = use_one_dropout

        self.output_dim = output_dim

        self.dropout = dropout

        self.vocab_size = vocab_size



        # Bert keras uses two segments for next-sentence classification task

        self.segment_emb = keras.layers.Embedding(2, output_dim, input_length=max_len,

                                                  name='SegmentEmbedding')



        self.trainable_pos_embedding = trainable_pos_embedding

        if not trainable_pos_embedding:

            self.pos_emb = keras.layers.Embedding(max_len, output_dim, trainable=False, input_length=max_len,

                                                  name='PositionEmbedding',

                                                  weights=[_get_pos_encoding_matrix(max_len, output_dim)])

        else:

            self.pos_emb = keras.layers.Embedding(max_len, output_dim, input_length=max_len, name='PositionEmbedding')



        self.token_emb = keras.layers.Embedding(vocab_size, output_dim, input_length=max_len, name='TokenEmbedding')

        self.embedding_dropout = keras.layers.Dropout(dropout, name='EmbeddingDropOut')

        self.add_embeddings = keras.layers.Add(name='AddEmbeddings')

        self.use_embedding_layer_norm = use_embedding_layer_norm

        if self.use_embedding_layer_norm:

            self.embedding_layer_norm = LayerNormalization(layer_norm_epsilon)

        else:

            self.embedding_layer_norm = None

        self.layer_norm_epsilon = layer_norm_epsilon



    def compute_output_shape(self, input_shape):

        return input_shape[0][0], input_shape[0][1], self.output_dim



    def get_config(self):

        config = {

            'max_len': self.max_len,

            'use_one_dropout': self.use_one_dropout,

            'output_dim': self.output_dim,

            'dropout': self.dropout,

            'vocab_size': self.vocab_size,

            'trainable_pos_embedding': self.trainable_pos_embedding,

            'embedding_layer_norm': self.use_embedding_layer_norm,

            'layer_norm_epsilon': self.layer_norm_epsilon

        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))



    def __call__(self, inputs, **kwargs):

        tokens, segment_ids, pos_ids = inputs

        segment_embedding = self.segment_emb(segment_ids)

        pos_embedding = self.pos_emb(pos_ids)

        token_embedding = self.token_emb(tokens)

        if self.use_one_dropout:

            summation = self.add_embeddings([segment_embedding, pos_embedding, token_embedding])

            if self.embedding_layer_norm:

                summation = self.embedding_layer_norm(summation)

            return self.embedding_dropout(summation)

        summation = self.add_embeddings(

            [self.embedding_dropout(segment_embedding), self.embedding_dropout(pos_embedding),

             self.embedding_dropout(token_embedding)])

        if self.embedding_layer_norm:

            summation = self.embedding_layer_norm(summation)

        return summation
class MultiHeadAttention(Layer):

    def __init__(self, n_head: int, n_state: int, attention_dropout: float, use_attn_mask: bool, neg_inf: float,

                 **kwargs) -> None:

        super().__init__(**kwargs)

        self.n_head = n_head

        self.n_state = n_state

        self.attention_dropout = attention_dropout

        self.use_attn_mask = use_attn_mask

        self.neg_inf = neg_inf



    def compute_output_shape(self, input_shape):

        x = input_shape[0] if self.use_attn_mask else input_shape

        return x[0], x[1], x[2] // 3



    def call(self, inputs, **kwargs):

        x = inputs[0] if self.use_attn_mask else inputs

        attn_mask = inputs[1] if self.use_attn_mask else None

        return multihead_attention(x, attn_mask, self.n_head, self.n_state, self.attention_dropout, self.neg_inf)



    def get_config(self):

        config = {

            'n_head': self.n_head,

            'n_state': self.n_state,

            'attention_dropout': self.attention_dropout,

            'use_attn_mask': self.use_attn_mask,

            'neg_inf': self.neg_inf,

        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))





class LayerNormalization(Layer):

    def __init__(self, eps: float = 1e-5, **kwargs) -> None:

        self.eps = eps

        super().__init__(**kwargs)



    def build(self, input_shape):

        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)

        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)

        super().build(input_shape)



    def call(self, x, **kwargs):

        u = K.mean(x, axis=-1, keepdims=True)

        s = K.mean(K.square(x - u), axis=-1, keepdims=True)

        z = (x - u) / K.sqrt(s + self.eps)

        return self.gamma * z + self.beta



    def compute_output_shape(self, input_shape):

        return input_shape



    def get_config(self):

        config = {

            'eps': self.eps,

        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))





class Gelu(Layer):

    def __init__(self, accurate: bool = False, **kwargs):

        super().__init__(**kwargs)

        self.accurate = accurate



    def call(self, inputs, **kwargs):

        if not self.accurate:

            return gelu(inputs)

        if K.backend() == 'tensorflow':

            erf = K.tf.erf

        else:

            erf = K.T.erf

        return inputs * 0.5 * (1.0 + erf(inputs / math.sqrt(2.0)))



    def compute_output_shape(self, input_shape):

        return input_shape



    def get_config(self):

        config = {

            'accurate': self.accurate,

        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))
class MultiHeadSelfAttention:

    def __init__(self, n_state: int, n_head: int, attention_dropout: float,

                 use_attn_mask: bool, layer_id: int, neg_inf: float) -> None:

        assert n_state % n_head == 0

        self.c_attn = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))

        self.attn = MultiHeadAttention(n_head, n_state, attention_dropout, use_attn_mask,

                                       neg_inf, name='layer_{}/self_attention'.format(layer_id))

        self.c_attn_proj = Conv1D(n_state, 1, name='layer_{}/c_attn_proj'.format(layer_id))



    def __call__(self, x, mask):

        output = self.c_attn(x)

        output = self.attn(output) if mask is None else self.attn([output, mask])

        return self.c_attn_proj(output)





class PositionWiseFF:

    def __init__(self, n_state: int, d_hid: int, layer_id: int, accurate_gelu: bool) -> None:

        self.c_fc = Conv1D(d_hid, 1, name='layer_{}/c_fc'.format(layer_id))

        self.activation = Gelu(accurate=accurate_gelu, name='layer_{}/gelu'.format(layer_id))

        self.c_ffn_proj = Conv1D(n_state, 1, name='layer_{}/c_ffn_proj'.format(layer_id))



    def __call__(self, x):

        output = self.activation(self.c_fc(x))

        return self.c_ffn_proj(output)





class EncoderLayer:

    def __init__(self, n_state: int, n_head: int, d_hid: int, residual_dropout: float, attention_dropout: float,

                 use_attn_mask: bool, layer_id: int, neg_inf: float, ln_epsilon: float, accurate_gelu: bool) -> None:

        self.attention = MultiHeadSelfAttention(n_state, n_head, attention_dropout, use_attn_mask, layer_id, neg_inf)

        self.drop1 = Dropout(residual_dropout, name='layer_{}/ln_1_drop'.format(layer_id))

        self.add1 = Add(name='layer_{}/ln_1_add'.format(layer_id))

        self.ln1 = LayerNormalization(ln_epsilon, name='layer_{}/ln_1'.format(layer_id))

        self.ffn = PositionWiseFF(n_state, d_hid, layer_id, accurate_gelu)

        self.drop2 = Dropout(residual_dropout, name='layer_{}/ln_2_drop'.format(layer_id))

        self.add2 = Add(name='layer_{}/ln_2_add'.format(layer_id))

        self.ln2 = LayerNormalization(ln_epsilon, name='layer_{}/ln_2'.format(layer_id))



    def __call__(self, x, mask):

        a = self.attention(x, mask)

        n = self.ln1(self.add1([x, self.drop1(a)]))

        f = self.ffn(n)

        return self.ln2(self.add2([n, self.drop2(f)]))





def create_transformer(embedding_dim: int = 768, embedding_dropout: float = 0.1, vocab_size: int = 30000,

                       max_len: int = 512, trainable_pos_embedding: bool = True, num_heads: int = 12,

                       num_layers: int = 12, attention_dropout: float = 0.1, use_one_embedding_dropout: bool = False,

                       d_hid: int = 768 * 4, residual_dropout: float = 0.1, use_attn_mask: bool = True,

                       embedding_layer_norm: bool = False, neg_inf: float = -1e9, layer_norm_epsilon: float = 1e-5,

                       accurate_gelu: bool = False) -> keras.Model:

    tokens = Input(batch_shape=(None, max_len), name='token_input', dtype='int32')

    segment_ids = Input(batch_shape=(None, max_len), name='segment_input', dtype='int32')

    pos_ids = Input(batch_shape=(None, max_len), name='position_input', dtype='int32')

    attn_mask = Input(batch_shape=(None, 1, max_len, max_len), name='attention_mask_input',

                      dtype=K.floatx()) if use_attn_mask else None

    inputs = [tokens, segment_ids, pos_ids]

    embedding_layer = BertEmbedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,

                                    use_one_embedding_dropout, embedding_layer_norm, layer_norm_epsilon)

    x = embedding_layer(inputs)

    for i in range(num_layers):

        x = EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout,

                         attention_dropout, use_attn_mask, i, neg_inf, layer_norm_epsilon, accurate_gelu)(x, attn_mask)

    if use_attn_mask:

        inputs.append(attn_mask)

    return keras.Model(inputs=inputs, outputs=[x], name='Transformer')
# coding=utf-8

# Copyright 2018 The Google AI Language Team Authors.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import copy

import json

import six

import tensorflow as tf





class BertConfig(object):

  """Configuration for `BertModel`."""



  def __init__(self,

               vocab_size,

               hidden_size=768,

               num_hidden_layers=12,

               num_attention_heads=12,

               intermediate_size=3072,

               hidden_act="gelu",

               hidden_dropout_prob=0.1,

               attention_probs_dropout_prob=0.1,

               max_position_embeddings=512,

               type_vocab_size=16,

               initializer_range=0.02):

    """Constructs BertConfig.

    Args:

      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.

      hidden_size: Size of the encoder layers and the pooler layer.

      num_hidden_layers: Number of hidden layers in the Transformer encoder.

      num_attention_heads: Number of attention heads for each attention layer in

        the Transformer encoder.

      intermediate_size: The size of the "intermediate" (i.e., feed-forward)

        layer in the Transformer encoder.

      hidden_act: The non-linear activation function (function or string) in the

        encoder and pooler.

      hidden_dropout_prob: The dropout probability for all fully connected

        layers in the embeddings, encoder, and pooler.

      attention_probs_dropout_prob: The dropout ratio for the attention

        probabilities.

      max_position_embeddings: The maximum sequence length that this model might

        ever be used with. Typically set this to something large just in case

        (e.g., 512 or 1024 or 2048).

      type_vocab_size: The vocabulary size of the `token_type_ids` passed into

        `BertModel`.

      initializer_range: The stdev of the truncated_normal_initializer for

        initializing all weight matrices.

    """

    self.vocab_size = vocab_size

    self.hidden_size = hidden_size

    self.num_hidden_layers = num_hidden_layers

    self.num_attention_heads = num_attention_heads

    self.hidden_act = hidden_act

    self.intermediate_size = intermediate_size

    self.hidden_dropout_prob = hidden_dropout_prob

    self.attention_probs_dropout_prob = attention_probs_dropout_prob

    self.max_position_embeddings = max_position_embeddings

    self.type_vocab_size = type_vocab_size

    self.initializer_range = initializer_range



  @classmethod

  def from_dict(cls, json_object):

    """Constructs a `BertConfig` from a Python dictionary of parameters."""

    config = BertConfig(vocab_size=None)

    for (key, value) in six.iteritems(json_object):

      config.__dict__[key] = value

    return config



  @classmethod

  def from_json_file(cls, json_file):

    """Constructs a `BertConfig` from a json file of parameters."""

    with tf.gfile.GFile(json_file, "r") as reader:

      text = reader.read()

    return cls.from_dict(json.loads(text))



  def to_dict(self):

    """Serializes this instance to a Python dictionary."""

    output = copy.deepcopy(self.__dict__)

    return output



  def to_json_string(self):

    """Serializes this instance to a JSON string."""

    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
def get_bert_weights_for_keras_model(check_point, max_len, model, tf_var_names):

    keras_weights = [np.zeros(w.shape) for w in model.weights]

    keras_weights_set = []



    for var_name, _ in tf_var_names:

        qkv, unsqueeze, w_id = _get_tf2keras_weights_name_mapping(var_name)

        if w_id is None:

            print('not mapped: ', var_name)  # TODO pooler, cls/predictions, cls/seq_relationship

        else:

            print(var_name, ' -> ', model.weights[w_id].name)

            keras_weights_set.append(w_id)

            keras_weight = keras_weights[w_id]

            tensorflow_weight = check_point.get_tensor(var_name)

            keras_weights[w_id] = _set_keras_weight_from_tf_weight(max_len, tensorflow_weight, keras_weight, qkv, unsqueeze, w_id)



    keras_layer_not_set = set(list(range(len(keras_weights)))) - set(keras_weights_set)

    assert len(keras_layer_not_set) == 0, 'Some weights were not set!'



    return keras_weights





def _set_keras_weight_from_tf_weight(max_len, tensorflow_weight, keras_weight, qkv, unsqueeze, w_id):

    if qkv is None:

        if w_id == 1:  # pos embedding

            keras_weight[:max_len, :] = tensorflow_weight[:max_len, :] if not unsqueeze else tensorflow_weight[None, :max_len, :]



        elif w_id == 2:  # word embedding

            keras_weight = tensorflow_weight

        else:

            keras_weight[:] = tensorflow_weight if not unsqueeze else tensorflow_weight[None, ...]

    else:

        p = {'q': 0, 'k': 1, 'v': 2}[qkv]

        if keras_weight.ndim == 3:

            dim_size = keras_weight.shape[1]

            keras_weight[0, :, p * dim_size:(p + 1) * dim_size] = tensorflow_weight if not unsqueeze else tensorflow_weight[None, ...]

        else:

            dim_size = keras_weight.shape[0] // 3

            keras_weight[p * dim_size:(p + 1) * dim_size] = tensorflow_weight



    return keras_weight





def _get_tf2keras_weights_name_mapping(var_name):

    w_id = None

    qkv = None

    unsqueeze = False



    var_name_splitted = var_name.split('/')

    if var_name_splitted[1] == 'embeddings':

        w_id = _get_embeddings_name(var_name_splitted)



    elif var_name_splitted[2].startswith('layer_'):

        qkv, unsqueeze, w_id = _get_layers_name(var_name_splitted)



    return qkv, unsqueeze, w_id





def _get_layers_name(var_name_splitted):

    first_vars_size = 5

    w_id = None

    qkv = None

    unsqueeze = False



    layer_number = int(var_name_splitted[2][len('layer_'):])

    if var_name_splitted[3] == 'attention':

        if var_name_splitted[-1] == 'beta':

            w_id = first_vars_size + layer_number * 12 + 5

        elif var_name_splitted[-1] == 'gamma':

            w_id = first_vars_size + layer_number * 12 + 4

        elif var_name_splitted[-2] == 'dense':

            if var_name_splitted[-1] == 'bias':

                w_id = first_vars_size + layer_number * 12 + 3

            elif var_name_splitted[-1] == 'kernel':

                w_id = first_vars_size + layer_number * 12 + 2

                unsqueeze = True

            else:

                raise ValueError()

        elif var_name_splitted[-2] == 'key' or var_name_splitted[-2] == 'query' or var_name_splitted[-2] == 'value':

            w_id = first_vars_size + layer_number * 12 + (0 if var_name_splitted[-1] == 'kernel' else 1)

            unsqueeze = var_name_splitted[-1] == 'kernel'

            qkv = var_name_splitted[-2][0]

        else:

            raise ValueError()

    elif var_name_splitted[3] == 'intermediate':

        if var_name_splitted[-1] == 'bias':

            w_id = first_vars_size + layer_number * 12 + 7

        elif var_name_splitted[-1] == 'kernel':

            w_id = first_vars_size + layer_number * 12 + 6

            unsqueeze = True

        else:

            raise ValueError()

    elif var_name_splitted[3] == 'output':

        if var_name_splitted[-1] == 'beta':

            w_id = first_vars_size + layer_number * 12 + 11

        elif var_name_splitted[-1] == 'gamma':

            w_id = first_vars_size + layer_number * 12 + 10

        elif var_name_splitted[-1] == 'bias':

            w_id = first_vars_size + layer_number * 12 + 9

        elif var_name_splitted[-1] == 'kernel':

            w_id = first_vars_size + layer_number * 12 + 8

            unsqueeze = True

        else:

            raise ValueError()

    return qkv, unsqueeze, w_id





def _get_embeddings_name(parts):

    n = parts[-1]

    if n == 'token_type_embeddings':

        w_id = 0

    elif n == 'position_embeddings':

        w_id = 1

    elif n == 'word_embeddings':

        w_id = 2

    elif n == 'gamma':

        w_id = 3

    elif n == 'beta':

        w_id = 4

    else:

        raise ValueError()

    return w_id
def load_google_bert(base_location: str = '../input/uncased_L-12_H-768_A-12/',

                     use_attn_mask: bool = True, max_len: int = 512) -> keras.Model:

    bert_config = BertConfig.from_json_file(base_location + 'bert_config.json')

    print(bert_config.__dict__)

    init_checkpoint = base_location + 'bert_model.ckpt'

    var_names = tf.train.list_variables(init_checkpoint)

    check_point = tf.train.load_checkpoint(init_checkpoint)

    model = create_transformer(embedding_layer_norm=True, neg_inf=-10000.0, use_attn_mask=use_attn_mask,

                               vocab_size=bert_config.vocab_size, accurate_gelu=True, layer_norm_epsilon=1e-12, max_len=max_len,

                               use_one_embedding_dropout=True, d_hid=bert_config.intermediate_size,

                               embedding_dim=bert_config.hidden_size, num_layers=bert_config.num_hidden_layers,

                               num_heads=bert_config.num_attention_heads,

                               residual_dropout=bert_config.hidden_dropout_prob,

                               attention_dropout=bert_config.attention_probs_dropout_prob)

    weights = get_bert_weights_for_keras_model(check_point, max_len, model, var_names)

    model.set_weights(weights)

    return model
BERT_PRETRAINED_DIR = '../input/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

g_bert = load_google_bert(base_location=BERT_PRETRAINED_DIR, use_attn_mask=False)

g_bert.summary()

g_bert.save('keras_bert_uncased_L-12_H-768_A-12.hdf5')