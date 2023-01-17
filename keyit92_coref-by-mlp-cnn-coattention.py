# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import gc

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.t
DATA_ROOT = '../input/'

GAP_DATA_FOLDER = os.path.join(DATA_ROOT, 'gap-coreference')

SUB_DATA_FOLDER = os.path.join(DATA_ROOT, 'gendered-pronoun-resolution')

FAST_TEXT_DATA_FOLDER = os.path.join(DATA_ROOT, 'fasttext-crawl-300d-2m')
test_df_path = os.path.join(GAP_DATA_FOLDER, 'gap-development.tsv')

train_df_path = os.path.join(GAP_DATA_FOLDER, 'gap-test.tsv')

dev_df_path = os.path.join(GAP_DATA_FOLDER, 'gap-validation.tsv')



train_df = pd.read_csv(train_df_path, sep='\t')

test_df = pd.read_csv(test_df_path, sep='\t')

dev_df = pd.read_csv(dev_df_path, sep='\t')



#pd.options.display.max_colwidth = 1000
train_df.head()
from spacy.lang.en import English

from spacy.pipeline import DependencyParser

import spacy

from nltk import Tree
nlp = spacy.load('en_core_web_lg')



def to_nltk_tree(node):

    if node.n_lefts + node.n_rights > 0:

        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])

    else:

        return node.orth_



def bs(list_, target_):

    lo, hi = 0, len(list_) -1

    

    while lo < hi:

        mid = lo + int((hi - lo) / 2)

        

        if target_ < list_[mid]:

            hi = mid

        elif target_ > list_[mid]:

            lo = mid + 1

        else:

            return mid + 1

    return lo



def _get_preceding_words(tokens, offset, k):

    start = offset - k

    

    precedings = [None] * max(0, 0-start)

    start = max(0, start)

    precedings += tokens[start: offset]

    

    return precedings



def _get_following_words(tokens, offset, k):

    end = offset + k

    

    followings = [None] * max(0, end - len(tokens))

    end = min(len(tokens), end)

    followings += tokens[offset: end]

    

    return followings

        



def extrac_embed_features_tokens(text, char_offset):

    doc = nlp(text)

    

    # char offset to token offset

    lens = [token.idx for token in doc]

    mention_offset = bs(lens, char_offset) - 1

    # mention_word

    mention = doc[mention_offset]

    

    # token offset to sentence offset

    lens = [len(sent) for sent in doc.sents]

    acc_lens = [len_ for len_ in lens]

    pre_len = 0

    for i in range(0, len(acc_lens)):

        pre_len += acc_lens[i]

        acc_lens[i] = pre_len

    sent_index = bs(acc_lens, mention_offset)

    # mention sentence

    sent = list(doc.sents)[sent_index]

    

    # dependency parent

    head = mention.head

    

    # last word and first word

    first_word, last_word = sent[0], sent[-2]

    

    assert mention_offset >= 0

    

    # two preceding words and two following words

    tokens = list(doc)

    precedings2 = _get_preceding_words(tokens, mention_offset, 2)

    followings2 = _get_following_words(tokens, mention_offset, 2)

    

    # five preceding words and five following words

    precedings5 = _get_preceding_words(tokens, mention_offset, 5)

    followings5 = _get_following_words(tokens, mention_offset, 5)

    

    # sentence words

    sent_tokens = [token for token in sent]

    

    return mention, head, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens
print("Texts: ")

text = u"Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline."

print(text)



print("\nDependency parsing trees: ")

doc = nlp(text)

[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]



print("\nFeatures:")

mention, parent, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens = extrac_embed_features_tokens(text, 274)

features = pd.Series([str(feature) for feature in (mention, parent, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens)], index=['mention', 'parent', 'first_word', 'last_word', 'precedings2', 'followings2', 'precedings5', 'followings5', 'sent_tokens'])

features
num_embed_features = 11

embed_dim = 300
def create_embedding_features(df, text_column, offset_column):

    text_offset_list = df[[text_column, offset_column]].values.tolist()

    num_features = num_embed_features

    

    embed_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features, embed_dim))

    for text_offset_index in range(len(text_offset_list)):

        text_offset = text_offset_list[text_offset_index]

        mention, parent, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens = extrac_embed_features_tokens(text_offset[0], text_offset[1])

        

        feature_index = 0

        embed_feature_matrix[text_offset_index, feature_index, :] = mention.vector

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = parent.vector

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = first_word.vector

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = last_word.vector

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index:feature_index+2, :] = np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings2])

        feature_index += len(precedings2)

        embed_feature_matrix[text_offset_index, feature_index:feature_index+2, :] = np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in followings2])

        feature_index += len(followings2)

        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings5]), axis=0)

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in followings5]), axis=0)

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(np.asarray([token.vector for token in sent_tokens]), axis=0) if len(sent_tokens) > 0 else np.zeros(embed_dim)

        feature_index += 1

    

    return embed_feature_matrix
def bs_(list_, target_):

    lo, hi = 0, len(list_) -1

    

    while lo < hi:

        mid = lo + int((hi - lo) / 2)

        

        if target_ < list_[mid]:

            hi = mid

        elif target_ > list_[mid]:

            lo = mid + 1

        else:

            return mid

    return lo



def ohe_dist(dist, buckets):

    idx = bs_(buckets, dist)

    oh = np.zeros(shape=(len(buckets),), dtype=np.float32)

    oh[idx] = 1

    

    return oh
def extrac_positional_features(text, char_offset1, char_offset2):

    doc = nlp(text)

    max_len = 64

    

    # char offset to token offset

    lens = [token.idx for token in doc]

    mention_offset1 = bs(lens, char_offset1) - 1

    mention_offset2 = bs(lens, char_offset2) - 1

    

    # token offset to sentence offset

    lens = [len(sent) for sent in doc.sents]

    acc_lens = [len_ for len_ in lens]

    pre_len = 0

    for i in range(0, len(acc_lens)):

        pre_len += acc_lens[i]

        acc_lens[i] = pre_len

    sent_index1 = bs(acc_lens, mention_offset1)

    sent_index2 = bs(acc_lens, mention_offset2)

    

    sent1 = list(doc.sents)[sent_index1]

    sent2 = list(doc.sents)[sent_index2]

    

    # buckets

    bucket_dist = [1, 2, 3, 4, 5, 8, 16, 32, 64]

    

    # relative distance

    dist = mention_offset2 - mention_offset1

    dist_oh = ohe_dist(dist, bucket_dist)

    

    # buckets

    bucket_pos = [0, 1, 2, 3, 4, 5, 8, 16, 32]

    

    # absolute position in the sentence

    sent_pos1 = mention_offset1 + 1

    if sent_index1 > 0:

        sent_pos1 = mention_offset1 - acc_lens[sent_index1-1]

    sent_pos_oh1 = ohe_dist(sent_pos1, bucket_pos)

    sent_pos_inv1 = len(sent1) - sent_pos1

    assert sent_pos_inv1 >= 0

    sent_pos_inv_oh1 = ohe_dist(sent_pos_inv1, bucket_pos)

    

    sent_pos2 = mention_offset2 + 1

    if sent_index2 > 0:

        sent_pos2 = mention_offset2 - acc_lens[sent_index2-1]

    sent_pos_oh2 = ohe_dist(sent_pos2, bucket_pos)

    sent_pos_inv2 = len(sent2) - sent_pos2

    if sent_pos_inv2 < 0:

        print(sent_pos_inv2)

        print(len(sent2))

        print(sent_pos2)

        raise ValueError

    sent_pos_inv_oh2 = ohe_dist(sent_pos_inv2, bucket_pos)

    

    sent_pos_ratio1 = sent_pos1 / len(sent1)

    sent_pos_ratio2 = sent_pos2 / len(sent2)

    

    return dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2
num_pos_features = 45
def create_dist_features(df, text_column, pronoun_offset_column, name_offset_column):

    text_offset_list = df[[text_column, pronoun_offset_column, name_offset_column]].values.tolist()

    num_features = num_pos_features

    

    pos_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features))

    for text_offset_index in range(len(text_offset_list)):

        text_offset = text_offset_list[text_offset_index]

        dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2 = extrac_positional_features(text_offset[0], text_offset[1], text_offset[2])

        

        feature_index = 0

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(dist_oh)] = np.asarray(dist_oh)

        feature_index += len(dist_oh)

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh1)] = np.asarray(sent_pos_oh1)

        feature_index += len(sent_pos_oh1)

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh2)] = np.asarray(sent_pos_oh2)

        feature_index += len(sent_pos_oh2)

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh1)] = np.asarray(sent_pos_inv_oh1)

        feature_index += len(sent_pos_inv_oh1)

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh2)] = np.asarray(sent_pos_inv_oh2)

        feature_index += len(sent_pos_inv_oh2)

    

    return pos_feature_matrix
p_emb_tra = create_embedding_features(train_df, 'Text', 'Pronoun-offset')

p_emb_dev = create_embedding_features(dev_df, 'Text', 'Pronoun-offset')

p_emb_test = create_embedding_features(test_df, 'Text', 'Pronoun-offset')



a_emb_tra = create_embedding_features(train_df, 'Text', 'A-offset')

a_emb_dev = create_embedding_features(dev_df, 'Text', 'A-offset')

a_emb_test = create_embedding_features(test_df, 'Text', 'A-offset')



b_emb_tra = create_embedding_features(train_df, 'Text', 'B-offset')

b_emb_dev = create_embedding_features(dev_df, 'Text', 'B-offset')

b_emb_test = create_embedding_features(test_df, 'Text', 'B-offset')



pa_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'A-offset')

pa_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'A-offset')

pa_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'A-offset')



pb_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'B-offset')

pb_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'B-offset')

pb_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'B-offset')
def _row_to_y(row):

    if row.loc['A-coref']:

        return 0

    if row.loc['B-coref']:

        return 1

    return 2



y_tra = train_df.apply(_row_to_y, axis=1)

y_dev = dev_df.apply(_row_to_y, axis=1)

y_test = test_df.apply(_row_to_y, axis=1)
X_train = [p_emb_tra, a_emb_tra, b_emb_tra, pa_pos_tra, pb_pos_tra]

X_dev = [p_emb_dev, a_emb_dev, b_emb_dev, pa_pos_dev, pb_pos_dev]

X_test = [p_emb_test, a_emb_test, b_emb_test, pa_pos_test, pb_pos_test]
import numpy as np

from keras import backend

from keras import layers

from keras import models
def build_mlp_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    model_dim, mlp_dim, mlp_depth=1, drop_out=0.5, return_customized_layers=False):

    """

    Create A Multi-Layer Perceptron Model.

    

    inputs: 

        embeddings: [batch, num_embed_feature, embed_dims] * 3 ## pronoun, A, B

        positional_features: [batch, num_pos_feature] * 2 ## pronoun-A, pronoun-B

        

    outputs: 

        [batch, num_classes] # in our case there should be 3 output classes: A, B, None

        

    :param output_dim: the output dimension size

    :param model_dim: rrn dimension size

    :param mlp_dim: the dimension size of fully connected layer

    :param mlp_depth: the depth of fully connected layers

    :param drop_out: dropout rate of fully connected layers

    :param return_customized_layers: boolean, default=False

        If True, return model and customized object dictionary, otherwise return model only

    :return: keras model

    """

    

    def _mlp_channel1(feature_dropout_layer, feature_map_layer, flatten_layer, x):

        x = feature_dropout_layer(x)

        x = feature_map_layer(x)

        x = flatten_layer(x)

        return x

    

    def _mlp_channel2(feature_map_layer, x):

        x = feature_map_layer(x)

        return x



    # inputs

    inputs1 = list()

    for fi in range(num_feature_channels1):

        inputs1.append(models.Input(shape=(num_features1, feature_dim1), dtype='float32', name='input1_' + str(fi)))

        

    inputs2 = list()

    for fi in range(num_feature_channels2):

        inputs2.append(models.Input(shape=(num_features2, ), dtype='float32', name='input2_' + str(fi)))

    

    # define feature map layers

    # MLP Layers

    feature_dropout_layer1 = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="input_dropout_layer"))

    feature_map_layer1 = layers.TimeDistributed(layers.Dense(model_dim, name="feature_map_layer1", activation="relu"))

    flatten_layer1 = layers.Flatten(name="feature_flatten_layer1")

    feature_map_layer2 = layers.Dense(model_dim, name="feature_map_layer2", activation="relu")

    

    x1 = [_mlp_channel1(feature_dropout_layer1, feature_map_layer1, flatten_layer1, input_) for input_ in inputs1]

    x2 = [_mlp_channel2(feature_map_layer2, input_) for input_ in inputs2]

    

    x = layers.Concatenate(axis=1, name="concate_layer")(x1+x2)

    

    # MLP Layers

    x = layers.BatchNormalization(name='batch_norm_layer')(x)

    x = layers.Dropout(rate=drop_out, name="dropout_layer")(x)

        

    for i in range(mlp_depth - 1):

        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)

        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)



    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)



    model = models.Model(inputs1 + inputs2, outputs)



    if return_customized_layers:

        return model, {}



    return model
def build_multi_channel_cnn_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    num_filters, filter_sizes, model_dim, mlp_dim, 

    mlp_depth=1, drop_out=0.5, pooling='max', padding='valid', return_customized_layers=False):

    """

    Create A Multi-Layer Perceptron Model.

    

    inputs: 

        embeddings: [batch, num_embed_feature, embed_dims] * 3 ## pronoun, A, B

        positional_features: [batch, num_pos_feature] * 2 ## pronoun-A, pronoun-B

        

    outputs: 

        [batch, num_classes] # in our case there should be 3 output classes: A, B, None

        

    :param output_dim: the output dimension size

    :param num_filters: list of integers

        The number of filters.

    :param filter_sizes: list of integers

        The kernel size.

    :param pooling: str, either 'max' or 'average'

        Pooling method.

    :param padding: One of "valid", "causal" or "same" (case-insensitive).

        Padding method.

    :param model_dim: rrn dimension size

    :param mlp_dim: the dimension size of fully connected layer

    :param mlp_depth: the depth of fully connected layers

    :param drop_out: dropout rate of fully connected layers

    :param return_customized_layers: boolean, default=False

        If True, return model and customized object dictionary, otherwise return model only

    :return: keras model

    """

    

    def _mlp_channel1(feature_dropout_layer, cnns, pools, concate_layer1, x):

        x = feature_dropout_layer(x)

        pooled_outputs = []

        for i in range(len(cnns)):

            conv = cnns[i](x)

            if pooling == 'max':

                conv = pools[i](conv)

            else:

                conv = pools[i](conv)

            pooled_outputs.append(conv)

        

        if len(cnns) == 1:

            x = conv

        else:

            x = concate_layer1(pooled_outputs)

        return x

    

    def _mlp_channel2(feature_map_layer, x):

        x = feature_map_layer(x)

        return x



    # inputs

    inputs1 = list()

    for fi in range(num_feature_channels1):

        inputs1.append(models.Input(shape=(num_features1, feature_dim1), dtype='float32', name='input1_' + str(fi)))

        

    inputs2 = list()

    for fi in range(num_feature_channels2):

        inputs2.append(models.Input(shape=(num_features2, ), dtype='float32', name='input2_' + str(fi)))

    

    # define feature map layers

    # CNN Layers

    cnns = []

    pools = []

    feature_dropout_layer1 = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="input_dropout_layer"))

    for i in range(len(filter_sizes)):

        cnns.append(layers.Conv1D(num_filters[i], kernel_size=filter_sizes[i], padding=padding, activation='relu', name="cc_layer1" + str(i)))

        if pooling == 'max':

            pools.append(layers.GlobalMaxPooling1D(name='global_pooling_layer1' + str(i)))

        else:

            pools.append(layers.GlobalAveragePooling1D(name='global_pooling_layer1' + str(i)))

    concate_layer1 = layers.Concatenate(name='concated_layer')

    

    feature_map_layer2 = layers.Dense(model_dim, name="feature_map_layer2", activation="relu")

    

    x1 = [_mlp_channel1(feature_dropout_layer1, cnns, pools, concate_layer1, input_) for input_ in inputs1]

    x2 = [_mlp_channel2(feature_map_layer2, input_) for input_ in inputs2]

    

    x = layers.Concatenate(axis=1, name="concate_layer")(x1+x2)

    

    # MLP Layers

    x = layers.BatchNormalization(name='batch_norm_layer')(x)

    x = layers.Dropout(rate=drop_out, name="dropout_layer")(x)

        

    for i in range(mlp_depth - 1):

        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)

        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)



    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)



    model = models.Model(inputs1 + inputs2, outputs)



    if return_customized_layers:

        return model, {}



    return model
from keras import initializers, regularizers, constraints, activations

from keras.engine import Layer

import keras.backend as K

from keras.layers import merge
def _dot_product(x, kernel):

    """

    Wrapper for dot product operation, in order to be compatible with both

    Theano and Tensorflow

    Args:

        x (): input

        kernel (): weights

    Returns:

    """

    if K.backend() == 'tensorflow':

        # todo: check that this is correct

        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

    else:

        return K.dot(x, kernel)

    

class RemappedCoAttentionWeight(merge._Merge):

    """

        Unnormalized Co-Attention operation for temporal data.

        Supports Masking.

        Follows the work of Ankur et al. [https://aclweb.org/anthology/D16-1244]

        "A Decomposable Attention Model for Natural Language Inference"

        # Input shape

            List of 2 3D tensor with shape: `(samples, steps1, features1)` and `(samples, steps2, features2)`.

        # Output shape

            3D tensor with shape: `(samples, steps1, step2)`.

        :param kwargs:

        """



    def __init__(self, model_size, activation='sigmoid',

                 W1_regularizer=None,  b1_regularizer=None,

                 W1_constraint=None, b1_constraint=None,

                 bias1=True, **kwargs):



        self.model_size = model_size

        self.init = initializers.get('glorot_uniform')



        self.W1_regularizer = regularizers.get(W1_regularizer)

        self.b1_regularizer = regularizers.get(b1_regularizer)



        self.W1_constraint = constraints.get(W1_constraint)

        self.b1_constraint = constraints.get(b1_constraint)



        self.bias1 = bias1

        self.activation = activations.get(activation)

        super(RemappedCoAttentionWeight, self).__init__(**kwargs)



    def build(self, input_shape):



        super(RemappedCoAttentionWeight, self).build(input_shape)

        if len(input_shape) != 2:

            raise ValueError("input must be a size two list which contains two tensors")



        shape1 = list(input_shape[0])

        shape2 = list(input_shape[1])



        self.W1 = self.add_weight((self.model_size, shape1[-1]),

                                 initializer=self.init,

                                 name='{}_W1'.format(self.name),

                                 regularizer=self.W1_regularizer,

                                 constraint=self.W1_constraint)



        self.W2 = self.W1



        if self.bias1:

            self.b1 = self.add_weight((self.model_size,),

                                     initializer='zero',

                                     name='{}_b1'.format(self.name),

                                     regularizer=self.b1_regularizer,

                                     constraint=self.b1_constraint)



        if self.bias1:

            self.b2 = self.b1



    def compute_mask(self, input, input_mask=None):

        # pass the mask to the next layers

        return input_mask



    def _merge_function(self, inputs):

        if len(inputs) != 2:

            raise ValueError('A `Subtract` layer should be called '

                             'on exactly 2 inputs')



        x1, x2 = inputs[0], inputs[1]



        # u = Wx + b

        u1 = _dot_product(x1, self.W1)

        if self.bias1:

            u1 += self.b1



        u2 = _dot_product(x2, self.W2)

        if self.bias1:

            u2 += self.b2



        # u = Activation(Wx + b)

        u1 = self.activation(u1)

        u2 = self.activation(u2)



        # atten = exp(u1 u2^T)

        atten = K.batch_dot(u1, u2, axes=[2, 2])

        atten = K.exp(atten)



        return atten



    def compute_output_shape(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:

            raise ValueError('A `Dot` layer should be called '

                             'on a list of 2 inputs.')

        shape1 = list(input_shape[0])

        shape2 = list(input_shape[1])



        if shape1[0] != shape2[0]:

            raise ValueError("batch size must be same")



        return shape1[0], shape1[1], shape2[1]



    def get_config(self):

        config = {

            'activation': self.activation,

            'model_size': self.model_size,

            'W1_regularizer': regularizers.serialize(self.W1_regularizer),

            'b1_regularizer': regularizers.serialize(self.b1_regularizer),

            'W1_constraint': constraints.serialize(self.W1_constraint),

            'b1_constraint': constraints.serialize(self.b1_constraint),

            'bias1': self.bias1,

        }

        base_config = super(RemappedCoAttentionWeight, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    

class FeatureNormalization(Layer):

    """

        Normalize feature along a specific axis.

        Supports Masking.



        # Input shape

            A ND tensor with shape: `(samples, feature1 ... featuresN).

        # Output shape

            ND tensor with shape: `(samples, feature1 ... featuresN)`.

        :param kwargs:

        """



    def __init__(self, axis=-1, **kwargs):



        self.axis = axis

        self.supports_masking = True

        super(FeatureNormalization, self).__init__(**kwargs)



    def build(self, input_shape):



        super(FeatureNormalization, self).build(input_shape)



    def compute_mask(self, input, input_mask=None):

        # don't pass the mask to the next layers

        return None



    def call(self, inputs, mask=None):

        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            # Cast the mask to floatX to avoid float64 upcasting in theano

            a = K.cast(mask, K.floatx()) * inputs

        else:

            a = inputs



        # in some cases especially in the early stages of training the sum may be almost zero

        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.

        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())

        a /= K.cast(K.sum(a, axis=self.axis, keepdims=True) + K.epsilon(), K.floatx())

        

        return a



    def compute_output_shape(self, input_shape):

        return input_shape



    def get_config(self):

        config = {

            'axis': self.axis

        }

        base_config = super(FeatureNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
def build_inter_coattention_cnn_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    num_filters, filter_sizes, atten_dim, model_dim, mlp_dim, 

    mlp_depth=1, drop_out=0.5, pooling='max', padding='valid', return_customized_layers=False):

    """

    Create A Multi-Layer Perceptron Model with Coattention Mechanism.

    

    inputs: 

        embeddings: [batch, num_embed_feature, embed_dims] * 3 ## pronoun, A, B

        positional_features: [batch, num_pos_feature] * 2 ## pronoun-A, pronoun-B

        

    outputs: 

        [batch, num_classes] # in our case there should be 3 output classes: A, B, None

        

    :param output_dim: the output dimension size

    :param model_dim: rrn dimension size

    :param mlp_dim: the dimension size of fully connected layer

    :param mlp_depth: the depth of fully connected layers

    :param drop_out: dropout rate of fully connected layers

    :param return_customized_layers: boolean, default=False

        If True, return model and customized object dictionary, otherwise return model only

    :return: keras model

    """

    

    def _mlp_channel1(feature_dropout_layer, x):

        #x = feature_dropout_layer(x)

        return x

    

    def _mlp_channel2(feature_map_layer, x):

        x = feature_map_layer(x)

        return x



    # inputs

    inputs1 = list()

    for fi in range(num_feature_channels1):

        inputs1.append(models.Input(shape=(num_features1, feature_dim1), dtype='float32', name='input1_' + str(fi)))

        

    inputs2 = list()

    for fi in range(num_feature_channels2):

        inputs2.append(models.Input(shape=(num_features2, ), dtype='float32', name='input2_' + str(fi)))

    

    # define feature map layers

    # MLP Layers

    feature_dropout_layer1 = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="input_dropout_layer"))

    feature_map_layer2 = layers.Dense(feature_dim1, name="feature_map_layer2", activation="relu")

    

    x1 = [_mlp_channel1(feature_dropout_layer1, input_) for input_ in inputs1]

    x2 = [_mlp_channel2(feature_map_layer2, input_) for input_ in inputs2]

    

    # From mention-pair embeddings

    reshape_layer = layers.Reshape((1, feature_dim1), name="reshape_layer")

    x2 = [reshape_layer(x2_) for x2_ in x2]

    pair1 = layers.Concatenate(axis=1, name="concate_pair1_layer")([x1[0], x1[1], x2[0]])

    pair2 = layers.Concatenate(axis=1, name="concate_pair2_layer")([x1[0], x1[2], x2[1]])

    

    coatten_layer = RemappedCoAttentionWeight(atten_dim, name="coattention_weights_layer")

    featnorm_layer1 = FeatureNormalization(name="normalized_coattention_weights_layer1", axis=1)

    featnorm_layer2 = FeatureNormalization(name="normalized_coattention_weights_layer2", axis=2)

    focus_layer1 = layers.Dot((1, 1), name="focus_layer1")

    focus_layer2 = layers.Dot((2, 1), name="focus_layer2")

    pair_layer1 = layers.Concatenate(axis=-1, name="pair_layer1")

    pair_layer2 = layers.Concatenate(axis=-1, name="pair_layer2")

    

    # attention

    attens = coatten_layer([pair1, pair2])

    attens1 = featnorm_layer1(attens)

    attens2 = featnorm_layer2(attens)

    # compare

    focus1 = focus_layer1([attens1, pair1])

    focus2 = focus_layer2([attens2, pair2])

    pair1 = pair_layer1([pair1, focus2])

    pair2 = pair_layer2([pair2, focus1])

    

    x = layers.Concatenate(axis=1, name="concate_layer")([pair1, pair2])

    x = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="pair_dropout_layer"))(x)

    x = layers.TimeDistributed(layers.Dense(mlp_dim, name="pair_feature_map_layer", activation="relu"))(x)

    x = layers.Flatten(name="pair_feature_flatten_layer1")(x)

    

#     pooled_outputs = []

#     for i in range(len(filter_sizes)):

#         conv = layers.Conv1D(num_filters[i], kernel_size=filter_sizes[i], padding=padding, activation='relu')(x)

#         if pooling == 'max':

#             conv = layers.GlobalMaxPooling1D(name='global_pooling_layer' + str(i))(conv)

#         else:

#             conv = layers.GlobalAveragePooling1D(name='global_pooling_layer' + str(i))(conv)

#         pooled_outputs.append(conv)

#     if len(pooled_outputs) > 1:

#         x = layers.Concatenate(name='concated_layer')(pooled_outputs)

#     else:

#         x = conv

    

    # MLP Layers

    x = layers.BatchNormalization(name='batch_norm_layer')(x)

    x = layers.Dropout(rate=drop_out, name="dropout_layer")(x)

        

    for i in range(mlp_depth - 1):

        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)

        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)



    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)



    model = models.Model(inputs1 + inputs2, outputs)



    if return_customized_layers:

        return model, {'RemappedCoAttentionWeight': RemappedCoAttentionWeight,

                       "FeatureNormalization": FeatureNormalization}



    return model
def build_intra_coattention_cnn_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    num_filters, filter_sizes, atten_dim, model_dim, mlp_dim, 

    mlp_depth=1, drop_out=0.5, pooling='max', padding='valid', return_customized_layers=False):

    """

    Create A Multi-Layer Perceptron Model with Coattention Mechanism.

    

    inputs: 

        embeddings: [batch, num_embed_feature, embed_dims] * 3 ## pronoun, A, B

        positional_features: [batch, num_pos_feature] * 2 ## pronoun-A, pronoun-B

        

    outputs: 

        [batch, num_classes] # in our case there should be 3 output classes: A, B, None

        

    :param output_dim: the output dimension size

    :param model_dim: rrn dimension size

    :param mlp_dim: the dimension size of fully connected layer

    :param mlp_depth: the depth of fully connected layers

    :param drop_out: dropout rate of fully connected layers

    :param return_customized_layers: boolean, default=False

        If True, return model and customized object dictionary, otherwise return model only

    :return: keras model

    """

    

    def _mlp_channel1(feature_dropout_layer, x):

        #x = feature_dropout_layer(x)

        return x

    

    def _mlp_channel2(feature_map_layer, x):

        x = feature_map_layer(x)

        return x

    

    def coatten_compare(

        feature_concat_layer, coatten_layer, 

        featnorm_layer1, featnorm_layer2, 

        focus_layer1, focus_layer2, 

        pair_layer1, pair_layer2, 

        mention, entity, mention_entity_feature):

        

        x1 = feature_concat_layer([entity, mention_entity_feature])

        x2 = feature_concat_layer([mention, mention_entity_feature])

        

        # attention

        attens = coatten_layer([x1, x2])

        attens1 = featnorm_layer1(attens)

        attens2 = featnorm_layer2(attens)

        # compare

        focus1 = focus_layer1([attens1, x1])

        focus2 = focus_layer2([attens2, x2])

        x1 = pair_layer1([x1, focus2])

        x2 = pair_layer2([x2, focus1])

        

        return x1, x2



    # inputs

    inputs1 = list()

    for fi in range(num_feature_channels1):

        inputs1.append(models.Input(shape=(num_features1, feature_dim1), dtype='float32', name='input1_' + str(fi)))

        

    inputs2 = list()

    for fi in range(num_feature_channels2):

        inputs2.append(models.Input(shape=(num_features2, ), dtype='float32', name='input2_' + str(fi)))

    

    # define feature map layers

    # MLP Layers

    feature_dropout_layer1 = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="input_dropout_layer"))

    feature_map_layer2 = layers.Dense(feature_dim1, name="feature_map_layer2", activation="relu")

    

    x1 = [_mlp_channel1(feature_dropout_layer1, input_) for input_ in inputs1]

    x2 = [_mlp_channel2(feature_map_layer2, input_) for input_ in inputs2]

    

    # From mention-pair embeddings

    reshape_layer = layers.Reshape((1, feature_dim1), name="reshape_layer")

    x2 = [reshape_layer(x2_) for x2_ in x2]

    

    feature_concat_layer = layers.Concatenate(axis=1, name="concate_pair_layer")

    coatten_layer = RemappedCoAttentionWeight(atten_dim, name="coattention_weights_layer")

    featnorm_layer1 = FeatureNormalization(name="normalized_coattention_weights_layer1", axis=1)

    featnorm_layer2 = FeatureNormalization(name="normalized_coattention_weights_layer2", axis=2)

    focus_layer1 = layers.Dot((1, 1), name="focus_layer1")

    focus_layer2 = layers.Dot((2, 1), name="focus_layer2")

    pair_layer1 = layers.Concatenate(axis=-1, name="pair_layer1")

    pair_layer2 = layers.Concatenate(axis=-1, name="pair_layer2")

    

    pairs = list()

    

    pairs += list(coatten_compare(

        feature_concat_layer, coatten_layer,

        featnorm_layer1, featnorm_layer2, 

        focus_layer1, focus_layer2, 

        pair_layer1, pair_layer2, 

        x1[0], x1[1], x2[0]))

    

    pairs += list(coatten_compare(

        feature_concat_layer, coatten_layer,

        featnorm_layer1, featnorm_layer2, 

        focus_layer1, focus_layer2, 

        pair_layer1, pair_layer2, 

        x1[0], x1[2], x2[1]))

    

    x = layers.Concatenate(axis=1, name="concate_layer")(pairs)

    x = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="pair_dropout_layer"))(x)

    x = layers.TimeDistributed(layers.Dense(mlp_dim, name="pair_feature_map_layer", activation="relu"))(x)

    x = layers.Flatten(name="pair_feature_flatten_layer1")(x)

    

#     pooled_outputs = []

#     for i in range(len(filter_sizes)):

#         conv = layers.Conv1D(num_filters[i], kernel_size=filter_sizes[i], padding=padding, activation='relu')(x)

#         if pooling == 'max':

#             conv = layers.GlobalMaxPooling1D(name='global_pooling_layer' + str(i))(conv)

#         else:

#             conv = layers.GlobalAveragePooling1D(name='global_pooling_layer' + str(i))(conv)

#         pooled_outputs.append(conv)

#     if len(pooled_outputs) > 1:

#         x = layers.Concatenate(name='concated_layer')(pooled_outputs)

#     else:

#         x = conv

    

    # MLP Layers

    x = layers.BatchNormalization(name='batch_norm_layer')(x)

    x = layers.Dropout(rate=drop_out, name="dropout_layer")(x)

        

    for i in range(mlp_depth - 1):

        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)

        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)



    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)



    model = models.Model(inputs1 + inputs2, outputs)



    if return_customized_layers:

        return model, {'RemappedCoAttentionWeight': RemappedCoAttentionWeight,

                       "FeatureNormalization": FeatureNormalization}



    return model
from keras import callbacks as kc

from keras import optimizers as ko

from keras import initializers, regularizers, constraints



import matplotlib.pyplot as plt

from IPython.display import SVG





histories = list()
num_feature_channels1 = 3

num_feature_channels2 = 2



num_embed_features = 11

num_features1 = num_embed_features

num_features2 = num_pos_features

feature_dim1 = embed_dim

output_dim = 3

model_dim = 10 

mlp_dim = 60

mlp_depth=1

drop_out=0.5

return_customized_layers=True



model, co_mlp = build_mlp_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    model_dim, mlp_dim, mlp_depth, drop_out, return_customized_layers

)
print(model.summary())
adam = ko.Nadam()

model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])



file_path = "best_mlp_model.hdf5"

check_point = kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")

early_stop = kc.EarlyStopping(monitor = "val_loss", mode = "min", patience=3)

history = model.fit(X_train, y_tra, batch_size=20, epochs=20, validation_data=(X_dev, y_dev), callbacks = [check_point, early_stop])



histories.append(np.min(np.asarray(history.history['val_loss'])))



del model, history

gc.collect()
num_feature_channels1 = 3

num_feature_channels2 = 2



num_embed_features = 11

num_features1 = num_embed_features

num_features2 = num_pos_features

feature_dim1 = embed_dim

output_dim = 3

model_dim = 10 

filter_sizes = [3, 5]

num_filters = [model_dim] * len(filter_sizes)

mlp_dim = 60

mlp_depth=1

pooling='max'

padding='valid'

drop_out=0.5

return_customized_layers=True



model, co_mccnn = build_multi_channel_cnn_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    num_filters, filter_sizes, model_dim, mlp_dim, mlp_depth, drop_out, pooling, padding, return_customized_layers

)
print(model.summary())
adam = ko.Nadam()

model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])



file_path = "best_mc_cnn_model.hdf5"

check_point = kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")

early_stop = kc.EarlyStopping(monitor = "val_loss", mode = "min", patience=3)

history = model.fit(X_train, y_tra, batch_size=20, epochs=20, validation_data=(X_dev, y_dev), callbacks = [check_point, early_stop])



histories.append(np.min(np.asarray(history.history['val_loss'])))



del model, history

gc.collect()
num_feature_channels1 = 3

num_feature_channels2 = 2



num_embed_features = 11

num_features1 = num_embed_features

num_features2 = num_pos_features

feature_dim1 = embed_dim

output_dim = 3

atten_dim = 10

model_dim = 10

filter_sizes = [1]

num_filters = [20] * len(filter_sizes)

mlp_dim = 5

mlp_depth=1

pooling='max'

padding='valid'

drop_out=0.5

return_customized_layers=True



model, co_cacnn = build_inter_coattention_cnn_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    num_filters, filter_sizes, atten_dim, model_dim, mlp_dim, mlp_depth, drop_out, pooling, padding, return_customized_layers

)
print(model.summary())
adam = ko.Nadam()

model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])



file_path = "best_coatt_cnn_model.hdf5"

check_point = kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")

early_stop = kc.EarlyStopping(monitor = "val_loss", mode = "min", patience=5)

history = model.fit(X_train, y_tra, batch_size=30, epochs=40, validation_data=(X_dev, y_dev), callbacks = [check_point, early_stop])



histories.append(np.min(np.asarray(history.history['val_loss'])))



del model, history

gc.collect()
num_feature_channels1 = 3

num_feature_channels2 = 2



num_embed_features = 11

num_features1 = num_embed_features

num_features2 = num_pos_features

feature_dim1 = embed_dim

output_dim = 3

atten_dim = 10

model_dim = 10

filter_sizes = [1]

num_filters = [20] * len(filter_sizes)

mlp_dim = 5

mlp_depth=1

pooling='max'

padding='valid'

drop_out=0.5

return_customized_layers=True



model, intra_co_cacnn = build_intra_coattention_cnn_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    num_filters, filter_sizes, atten_dim, model_dim, mlp_dim, mlp_depth, drop_out, pooling, padding, return_customized_layers

)
print(model.summary())
adam = ko.Nadam()

model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])



file_path = "best_intra_coatt_cnn_model.hdf5"

check_point = kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")

early_stop = kc.EarlyStopping(monitor = "val_loss", mode = "min", patience=5)

history = model.fit(X_train, y_tra, batch_size=30, epochs=40, validation_data=(X_dev, y_dev), callbacks = [check_point, early_stop])



histories.append(np.min(np.asarray(history.history['val_loss'])))



del model, history

gc.collect()
model_paths = [

    "best_mlp_model.hdf5",

    "best_mc_cnn_model.hdf5",

    "best_coatt_cnn_model.hdf5",

    "best_intra_coatt_cnn_model.hdf5"

]



cls_ =[

    co_mlp, co_mccnn, co_cacnn, intra_co_cacnn

]



print("load best model: " + str(model_paths[np.argmin(histories)]))

model = models.load_model(

    model_paths[np.argmin(histories)], cls_[np.argmin(histories)])
y_preds = model.predict(X_test, batch_size = 1024, verbose = 1)



sub_df_path = os.path.join(SUB_DATA_FOLDER, 'sample_submission_stage_1.csv')

sub_df = pd.read_csv(sub_df_path)

sub_df.loc[:, 'A'] = pd.Series(y_preds[:, 0])

sub_df.loc[:, 'B'] = pd.Series(y_preds[:, 1])

sub_df.loc[:, 'NEITHER'] = pd.Series(y_preds[:, 2])



sub_df.head()
sub_df.to_csv("submission.csv", index=False)