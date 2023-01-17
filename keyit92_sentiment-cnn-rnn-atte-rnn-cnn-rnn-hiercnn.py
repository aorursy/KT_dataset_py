import pandas as pd

import numpy as np

import os



import gc
DATA_ROOT = '../input/'

ORIGINAL_DATA_FOLDER = os.path.join(DATA_ROOT, 'movie-review-sentiment-analysis-kernels-only')

TMP_DATA_FOLDER = os.path.join(DATA_ROOT, 'kaggle_review_sentiment_tmp_data')
train_data_path = os.path.join(ORIGINAL_DATA_FOLDER, 'train.tsv')

test_data_path = os.path.join(ORIGINAL_DATA_FOLDER, 'test.tsv')

sub_data_path = os.path.join(ORIGINAL_DATA_FOLDER, 'sampleSubmission.csv')



train_df = pd.read_csv(train_data_path, sep="\t")

test_df = pd.read_csv(test_data_path, sep="\t")

sub_df = pd.read_csv(sub_data_path, sep=",")
import seaborn as sns

from sklearn.feature_extraction import text as sktext
train_df.head()
test_df.head()
sub_df.head()
overlapped = pd.merge(train_df[["Phrase", "Sentiment"]], test_df, on="Phrase", how="inner")

overlap_boolean_mask_test = test_df['Phrase'].isin(overlapped['Phrase'])
print("training data phrase length distribution")

sns.distplot(train_df['Phrase'].map(lambda ele: len(ele)), kde_kws={"label": "train"})



print("testing data phrase length distribution")

sns.distplot(test_df[~overlap_boolean_mask_test]['Phrase'].map(lambda ele: len(ele)), kde_kws={"label": "test"})
print("training and testing data sentences hist:")

sns.distplot(train_df['SentenceId'], kde_kws={"label": "train"})

sns.distplot(test_df['SentenceId'], kde_kws={"label": "test"})
print("The number of overlapped SentenceId between training and testing data:")

train_overlapped_sentence_id_df = train_df[train_df['SentenceId'].isin(test_df['SentenceId'])]

print(train_overlapped_sentence_id_df.shape[0])



del train_overlapped_sentence_id_df

gc.collect()
pd.options.display.max_colwidth = 250

print("Example of sentence and phrases: ")



sample_sentence_id = train_df.sample(1)['SentenceId'].values[0]

sample_sentence_group_df = train_df[train_df['SentenceId'] == sample_sentence_id]

sample_sentence_group_df
import nltk

import gensim

import operator 

from keras.preprocessing import text as ktext
def build_vocab(texts):

    tk = ktext.Tokenizer(lower = True, filters='')

    tk.fit_on_texts(texts)

    return tk.word_counts



def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    for word in vocab.keys():

        if word in embeddings_index:

            known_words[word] = vocab[word]

            continue

        unknown_words[word] = vocab[word]



    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    num_known_words = np.sum(np.asarray(list(known_words.values())))

    num_unknown_words = np.sum(np.asarray(list(unknown_words.values())))

    print('Found embeddings for  {:.3%} of all text'.format(float(num_known_words) / (num_known_words + num_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
texts = list(train_df['Phrase'].values) + list(test_df['Phrase'].values)

vocab = build_vocab(texts)
def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr[:len(arr)-1], dtype='float32')

    

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>15)

        

    return embeddings_index
pretrained_w2v_path = os.path.join(DATA_ROOT, "nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin")

w2v_google = gensim.models.KeyedVectors.load_word2vec_format(pretrained_w2v_path, binary=True).wv



pretrained_w2v_path = os.path.join(DATA_ROOT, "fasttext-crawl-300d-2m/crawl-300d-2M.vec")

w2v_fasttext = load_embed(pretrained_w2v_path)
print("google")

unknown_vocab = check_coverage(vocab, w2v_google)

print("unknown vocabulary:")

print(unknown_vocab[:50])



print("\n")



print("fast text")

unknown_vocab = check_coverage(vocab, w2v_fasttext)

print("unknown vocabulary:")

print(unknown_vocab[:50])
w2v = w2v_fasttext

# del w2v_google, w2v_fasttext, texts, vocab

# gc.collect()
from keras.preprocessing import sequence

import gensim

from sklearn import preprocessing as skp
max_len = 50

embed_size = 300

max_features = 30000
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", "n't": "not", "'ve": "have"}
def known_contractions(embed):

    known = []

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known
known_contract_list = known_contractions(w2v)

print(known_contract_list)
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
train_df.loc[:, 'Phrase'] = train_df['Phrase'].map(lambda text: clean_contractions(text, contraction_mapping))

test_df.loc[:, 'Phrase'] = test_df['Phrase'].map(lambda text: clean_contractions(text, contraction_mapping))
full_text = list(train_df['Phrase'].values) + list(test_df['Phrase'].values)

vocab = build_vocab(full_text)

check_coverage(vocab, w2v)

print("")
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown



def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
print(unknown_punct(w2v, punct))
train_df.loc[:, 'Phrase'] = train_df['Phrase'].map(lambda text: clean_special_chars(text, punct, punct_mapping))

test_df.loc[:, 'Phrase'] = test_df['Phrase'].map(lambda text: clean_special_chars(text, punct, punct_mapping))
full_text = list(train_df['Phrase'].values) + list(test_df['Phrase'].values)

vocab = build_vocab(full_text)

unknown_vocab = check_coverage(vocab, w2v)
def map_unknown_token(text, dst_token, unknown_vocab_set):

#     token_list = []

#     for t in text.split(" "):

#         if t in unknown_vocab_set:

#             token_list.append(dst_token)

#         else:

#             token_list.append(t)

    

#     return " ".join(token_list)

    return' '.join([dst_token if t.lower() in unknown_vocab_set else t for t in text.split(" ")])
unknown_vocab_set = set(list(map(

    lambda unknown_vocab_tuple: unknown_vocab_tuple[0],

    unknown_vocab

)))

train_df.loc[:, 'Phrase'] = train_df['Phrase'].map(lambda ele: map_unknown_token(ele, "[ name ]", unknown_vocab_set))

test_df.loc[:, 'Phrase'] = test_df['Phrase'].map(lambda ele: map_unknown_token(ele, "[ name ]", unknown_vocab_set))
full_text = list(train_df['Phrase'].values) + list(test_df['Phrase'].values)

vocab = build_vocab(full_text)

unknown_vocab = check_coverage(vocab, w2v)

print(unknown_vocab)
full_text = list(train_df['Phrase'].values) + list(test_df[~overlap_boolean_mask_test]['Phrase'].values)



tk = ktext.Tokenizer(lower = True, filters='')

tk.fit_on_texts(full_text)

train_tokenized = tk.texts_to_sequences(train_df['Phrase'])

test_tokenized = tk.texts_to_sequences(test_df[~overlap_boolean_mask_test]['Phrase'])



X_train = sequence.pad_sequences(train_tokenized, maxlen = max_len)

X_test = sequence.pad_sequences(test_tokenized, maxlen = max_len)
word_index = tk.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words + 1, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = None

    if word in w2v:

        embedding_vector = w2v[word]

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

del w2v

gc.collect()
y_train = train_df['Sentiment']



led = skp.LabelEncoder()

led.fit(y_train.values)



y_train = led.transform(y_train.values)
import tensorflow as tf



from keras import callbacks as kc

from keras import optimizers as ko

from keras import initializers, regularizers, constraints

from keras.engine import Layer

import keras.backend as K
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

    

    

class AttentionWeight(Layer):

    """

        This code is a modified version of cbaziotis implementation:  GithubGist cbaziotis/AttentionWithContext.py

        Attention operation, with a context/query vector, for temporal data.

        Supports Masking.

        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]

        "Hierarchical Attention Networks for Document Classification"

        by using a context vector to assist the attention

        # Input shape

            3D tensor with shape: `(samples, steps, features)`.

        # Output shape

            2D tensor with shape: `(samples, steps)`.

        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

        The dimensions are inferred based on the output shape of the RNN.

        Example:

            model.add(LSTM(64, return_sequences=True))

            model.add(AttentionWeight())

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

        super(AttentionWeight, self).__init__(**kwargs)



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



        super(AttentionWeight, self).build(input_shape)



    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers

        return None



    def call(self, x, mask=None):

        uit = _dot_product(x, self.W)



        if self.bias:

            uit += self.b



        uit = K.tanh(uit)

        ait = _dot_product(uit, self.u)



        a = K.exp(ait)



        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            # Cast the mask to floatX to avoid float64 upcasting in theano

            a *= K.cast(mask, K.floatx())



        # in some cases especially in the early stages of training the sum may be almost zero

        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.

        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        return a



    def compute_output_shape(self, input_shape):

        return input_shape[0], input_shape[1]



    def get_config(self):

        config = {

            'W_regularizer': regularizers.serialize(self.W_regularizer),

            'u_regularizer': regularizers.serialize(self.u_regularizer),

            'b_regularizer': regularizers.serialize(self.b_regularizer),

            'W_constraint': constraints.serialize(self.W_constraint),

            'u_constraint': constraints.serialize(self.u_constraint),

            'b_constraint': constraints.serialize(self.b_constraint),

            'bias': self.bias

        }

        base_config = super(AttentionWeight, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
def is_integer(val):

    return isinstance(val, (int, np.int_))



def predict(keras_model, x, learning_phase=0):



    if isinstance(keras_model.input, list):

        f = backend.function(

            keras_model.input + [backend.learning_phase()],

            [keras_model.output, ]

        )

        y = f(tuple(x) + (learning_phase,))[0]

    else:

        f = backend.function(

            [keras_model.input, backend.learning_phase()],

            [keras_model.output, ]

        )

        y = f((x, learning_phase))[0]

    return y

    



def build_birnn_attention_model(

        voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, 

        item_embedding=None, rnn_depth=1, mlp_depth=1, num_att_channel=1,

        drop_out=0.5, rnn_drop_out=0., rnn_state_drop_out=0.,

        trainable_embedding=False, gpu=False, return_customized_layers=False):

    """

    Create A Bidirectional Attention Model.



    :param voca_dim: vocabulary dimension size.

    :param time_steps: the length of input

    :param output_dim: the output dimension size

    :param rnn_dim: rrn dimension size

    :param mlp_dim: the dimension size of fully connected layer

    :param item_embedding: integer, numpy 2D array, or None (default=None)

        If item_embedding is a integer, connect a randomly initialized embedding matrix to the input tensor.

        If item_embedding is a matrix, this matrix will be used as the embedding matrix.

        If item_embedding is None, then connect input tensor to RNN layer directly.

    :param rnn_depth: rnn depth

    :param mlp_depth: the depth of fully connected layers

    :param num_att_channel: the number of attention channels, this can be used to mimic multi-head attention mechanism

    :param drop_out: dropout rate of fully connected layers

    :param rnn_drop_out: dropout rate of rnn layers

    :param rnn_state_drop_out: dropout rate of rnn state tensor

    :param trainable_embedding: boolean

    :param gpu: boolean, default=False

        If True, CuDNNLSTM is used instead of LSTM for RNN layer.

    :param return_customized_layers: boolean, default=False

        If True, return model and customized object dictionary, otherwise return model only

    :return: keras model

    """

    

    if item_embedding is not None:

        inputs = models.Input(shape=(time_steps,), dtype='int32', name='input0')

        x = inputs



        # item embedding

        if isinstance(item_embedding, np.ndarray):

            assert voca_dim == item_embedding.shape[0]

            x = layers.Embedding(

                voca_dim, item_embedding.shape[1], input_length=time_steps,

                weights=[item_embedding, ], trainable=trainable_embedding,

                mask_zero=False, name='embedding_layer0'

            )(x)

        elif utils.is_integer(item_embedding):

            x = layers.Embedding(

                voca_dim, item_embedding, input_length=time_steps,

                trainable=trainable_embedding,

                mask_zero=False, name='embedding_layer0'

            )(x)

        else:

            raise ValueError("item_embedding must be either integer or numpy matrix")

    else:

        inputs = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='input0')

        x = inputs

    

    x = layers.SpatialDropout1D(rnn_drop_out, name='rnn_spatial_droutout_layer')(x)



    if gpu:

        # rnn encoding

        for i in range(rnn_depth):

            x = layers.Bidirectional(

                layers.CuDNNLSTM(rnn_dim, return_sequences=True),

                name='bi_lstm_layer' + str(i))(x)

            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)

            x = layers.Dropout(rate=rnn_drop_out, name="rnn_dropout_layer" + str(i))(x)

    else:

        # rnn encoding

        for i in range(rnn_depth):

            x = layers.Bidirectional(

                layers.LSTM(rnn_dim, return_sequences=True, dropout=rnn_drop_out, recurrent_dropout=rnn_state_drop_out),

                name='bi_lstm_layer' + str(i))(x)

            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)



    # attention

    attention_heads = []

    x_per = layers.Permute((2, 1), name='permuted_attention_x')(x)

    for h in range(max(1, num_att_channel)):

        attention = AttentionWeight(name="attention_weights_layer" + str(h))(x)

        xx = layers.Dot([2, 1], name='focus_head' + str(h) + '_layer0')([x_per, attention])

        attention_heads.append(xx)



    if num_att_channel > 1:

        x = layers.Concatenate(name='focus_layer0')(attention_heads)

    else:

        x = attention_heads[0]



    x = layers.BatchNormalization(name='focused_batch_norm_layer')(x)

    x = layers.Dropout(rate=rnn_drop_out, name="focused_dropout_layer")(x)



    # MLP Layers

    for i in range(mlp_depth - 1):

        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)

        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)



    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)



    model = models.Model(inputs, outputs)



    if return_customized_layers:

        return model, {'AttentionWeight': AttentionWeight}

    return model





def build_cnn_model(

        voca_dim, time_steps, output_dim, mlp_dim, num_filters, filter_sizes,

        item_embedding=None, mlp_depth=1,

        drop_out=0.5, cnn_drop_out=0.5, pooling='max', padding='valid',

        trainable_embedding=False, return_customized_layers=False):

    """

    Create A CNN Model.



    :param voca_dim: vocabulary dimension size.

    :param time_steps: the length of input

    :param output_dim: the output dimension size

    :param num_filters: list of integers

        The number of filters.

    :param filter_sizes: list of integers

        The kernel size.

    :param mlp_dim: the dimension size of fully connected layer

    :param item_embedding: integer, numpy 2D array, or None (default=None)

        If item_embedding is a integer, connect a randomly initialized embedding matrix to the input tensor.

        If item_embedding is a matrix, this matrix will be used as the embedding matrix.

        If item_embedding is None, then connect input tensor to RNN layer directly.

    :param mlp_depth: the depth of fully connected layers

    :param drop_out: dropout rate of fully connected layers

    :param cnn_drop_out: dropout rate of between cnn layer and fully connected layers

    :param pooling: str, either 'max' or 'average'

        Pooling method.

    :param padding: One of "valid", "causal" or "same" (case-insensitive).

        Padding method.

    :param trainable_embedding: boolean

    :param return_customized_layers: boolean, default=False

        If True, return model and customized object dictionary, otherwise return model only

    :return: keras model

    """



    if item_embedding is not None:

        inputs = models.Input(shape=(time_steps,), dtype='int32', name='input0')

        x = inputs



        # item embedding

        if isinstance(item_embedding, np.ndarray):

            assert voca_dim == item_embedding.shape[0]

            x = layers.Embedding(

                voca_dim, item_embedding.shape[1], input_length=time_steps,

                weights=[item_embedding, ], trainable=trainable_embedding,

                mask_zero=False, name='embedding_layer0'

            )(x)

        elif utils.is_integer(item_embedding):

            x = layers.Embedding(

                voca_dim, item_embedding, input_length=time_steps,

                trainable=trainable_embedding,

                mask_zero=False, name='embedding_layer0'

            )(x)

        else:

            raise ValueError("item_embedding must be either integer or numpy matrix")

    else:

        inputs = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='input0')

        x = inputs

    

    x = layers.SpatialDropout1D(cnn_drop_out, name='cnn_spatial_droutout_layer')(x)



    pooled_outputs = []

    for i in range(len(filter_sizes)):

        conv = layers.Conv1D(num_filters[i], kernel_size=filter_sizes[i], padding=padding, activation='relu')(x)

        if pooling == 'max':

            conv = layers.GlobalMaxPooling1D(name='global_pooling_layer' + str(i))(conv)

        else:

            conv = layers.GlobalAveragePooling1D(name='global_pooling_layer' + str(i))(conv)

        pooled_outputs.append(conv)



    x = layers.Concatenate(name='concated_layer')(pooled_outputs)

    x = layers.Dropout(cnn_drop_out, name='conv_dropout_layer')(x)

    x = layers.BatchNormalization(name="batch_norm_layer")(x)



    # MLP Layers

    for i in range(mlp_depth - 1):

        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)

        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)



    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)



    model = models.Model(inputs, outputs)



    if return_customized_layers:

        return model, dict()



    return model





def build_birnn_cnn_model(

        voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, num_filters, filter_sizes,

        item_embedding=None, rnn_depth=1, mlp_depth=1,

        drop_out=0.5, rnn_drop_out=0.5, rnn_state_drop_out=0.5, cnn_drop_out=0.5, pooling='max', padding='valid',

        trainable_embedding=False, gpu=False, return_customized_layers=False):

    """

    Create A Bidirectional CNN Model.



    :param voca_dim: vocabulary dimension size.

    :param time_steps: the length of input

    :param output_dim: the output dimension size

    :param rnn_dim: rrn dimension size

    :param num_filters: list of integers

        The number of filters.

    :param filter_sizes: list of integers

        The kernel size.

    :param mlp_dim: the dimension size of fully connected layer

    :param item_embedding: integer, numpy 2D array, or None (default=None)

        If item_embedding is a integer, connect a randomly initialized embedding matrix to the input tensor.

        If item_embedding is a matrix, this matrix will be used as the embedding matrix.

        If item_embedding is None, then connect input tensor to RNN layer directly.

    :param rnn_depth: rnn depth

    :param mlp_depth: the depth of fully connected layers

    :param num_att_channel: the number of attention channels, this can be used to mimic multi-head attention mechanism

    :param drop_out: dropout rate of fully connected layers

    :param rnn_drop_out: dropout rate of rnn layers

    :param rnn_state_drop_out: dropout rate of rnn state tensor

    :param cnn_drop_out: dropout rate of between cnn layer and fully connected layers

    :param pooling: str, either 'max' or 'average'

        Pooling method.

    :param padding: One of "valid", "causal" or "same" (case-insensitive).

        Padding method.

    :param trainable_embedding: boolean

    :param gpu: boolean, default=False

        If True, CuDNNLSTM is used instead of LSTM for RNN layer.

    :param return_customized_layers: boolean, default=False

        If True, return model and customized object dictionary, otherwise return model only

    :return: keras model

    """



    if item_embedding is not None:

        inputs = models.Input(shape=(time_steps,), dtype='int32', name='input0')

        x = inputs



        # item embedding

        if isinstance(item_embedding, np.ndarray):

            assert voca_dim == item_embedding.shape[0]

            x = layers.Embedding(

                voca_dim, item_embedding.shape[1], input_length=time_steps,

                weights=[item_embedding, ], trainable=trainable_embedding,

                mask_zero=False, name='embedding_layer0'

            )(x)

        elif utils.is_integer(item_embedding):

            x = layers.Embedding(

                voca_dim, item_embedding, input_length=time_steps,

                trainable=trainable_embedding,

                mask_zero=False, name='embedding_layer0'

            )(x)

        else:

            raise ValueError("item_embedding must be either integer or numpy matrix")

    else:

        inputs = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='input0')

        x = inputs

        

    x = layers.SpatialDropout1D(rnn_drop_out, name='rnn_spatial_droutout_layer')(x)



    if gpu:

        # rnn encoding

        for i in range(rnn_depth):

            x = layers.Bidirectional(

                layers.CuDNNLSTM(rnn_dim, return_sequences=True),

                name='bi_lstm_layer' + str(i))(x)

            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)

            x = layers.Dropout(rate=rnn_drop_out, name="rnn_dropout_layer" + str(i))(x)

    else:

        # rnn encoding

        for i in range(rnn_depth):

            x = layers.Bidirectional(

                layers.LSTM(rnn_dim, return_sequences=True, dropout=rnn_drop_out, recurrent_dropout=rnn_state_drop_out),

                name='bi_lstm_layer' + str(i))(x)

            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)



    pooled_outputs = []

    for i in range(len(filter_sizes)):

        conv = layers.Conv1D(num_filters[i], kernel_size=filter_sizes[i], padding=padding, activation='relu')(x)

        if pooling == 'max':

            conv = layers.GlobalMaxPooling1D(name='global_pooling_layer' + str(i))(conv)

        else:

            conv = layers.GlobalAveragePooling1D(name='global_pooling_layer' + str(i))(conv)

        pooled_outputs.append(conv)



    x = layers.Concatenate(name='concated_layer')(pooled_outputs)

    x = layers.BatchNormalization(name="batch_norm_layer")(x)

    x = layers.Dropout(cnn_drop_out, name='conv_dropout_layer')(x)



    # MLP Layers

    for i in range(mlp_depth - 1):

        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)

        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)



    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)



    model = models.Model(inputs, outputs)



    if return_customized_layers:

        return model, dict()



    return model



def build_birnn_hierarchy_cnn_model(

        voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, num_filters, filter_sizes, 

        dilation_rates=1, strides=1,

        item_embedding=None, rnn_depth=1, mlp_depth=1,

        drop_out=0.5, rnn_drop_out=0.5, rnn_state_drop_out=0.5, cnn_drop_out=0.5, pooling='max', padding='valid',

        trainable_embedding=False, gpu=False, return_customized_layers=False):

    """

    Create A Bidirectional CNN Model.



    :param voca_dim: vocabulary dimension size.

    :param time_steps: the length of input

    :param output_dim: the output dimension size

    :param rnn_dim: rrn dimension size

    :param num_filters: list of integers

        The number of filters.

    :param filter_sizes: list of integers

        The kernel size.

    :param mlp_dim: the dimension size of fully connected layer

    :param item_embedding: integer, numpy 2D array, or None (default=None)

        If item_embedding is a integer, connect a randomly initialized embedding matrix to the input tensor.

        If item_embedding is a matrix, this matrix will be used as the embedding matrix.

        If item_embedding is None, then connect input tensor to RNN layer directly.

    :param rnn_depth: rnn depth

    :param mlp_depth: the depth of fully connected layers

    :param num_att_channel: the number of attention channels, this can be used to mimic multi-head attention mechanism

    :param drop_out: dropout rate of fully connected layers

    :param rnn_drop_out: dropout rate of rnn layers

    :param rnn_state_drop_out: dropout rate of rnn state tensor

    :param cnn_drop_out: dropout rate of between cnn layer and fully connected layers

    :param pooling: str, either 'max' or 'average'

        Pooling method.

    :param padding: One of "valid", "causal" or "same" (case-insensitive).

        Padding method.

    :param trainable_embedding: boolean

    :param gpu: boolean, default=False

        If True, CuDNNLSTM is used instead of LSTM for RNN layer.

    :param return_customized_layers: boolean, default=False

        If True, return model and customized object dictionary, otherwise return model only

    :return: keras model

    """



    if item_embedding is not None:

        inputs = models.Input(shape=(time_steps,), dtype='int32', name='input0')

        x = inputs



        # item embedding

        if isinstance(item_embedding, np.ndarray):

            assert voca_dim == item_embedding.shape[0]

            x = layers.Embedding(

                voca_dim, item_embedding.shape[1], input_length=time_steps,

                weights=[item_embedding, ], trainable=trainable_embedding,

                mask_zero=False, name='embedding_layer0'

            )(x)

        elif utils.is_integer(item_embedding):

            x = layers.Embedding(

                voca_dim, item_embedding, input_length=time_steps,

                trainable=trainable_embedding,

                mask_zero=False, name='embedding_layer0'

            )(x)

        else:

            raise ValueError("item_embedding must be either integer or numpy matrix")

    else:

        inputs = models.Input(shape=(time_steps, voca_dim), dtype='float32', name='input0')

        x = inputs

        

    x = layers.SpatialDropout1D(rnn_drop_out, name='rnn_spatial_droutout_layer')(x)



    if gpu:

        # rnn encoding

        for i in range(rnn_depth):

            x = layers.Bidirectional(

                layers.CuDNNLSTM(rnn_dim, return_sequences=True),

                name='bi_lstm_layer' + str(i))(x)

            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)

            x = layers.Dropout(rate=rnn_drop_out, name="rnn_dropout_layer" + str(i))(x)

    else:

        # rnn encoding

        for i in range(rnn_depth):

            x = layers.Bidirectional(

                layers.LSTM(rnn_dim, return_sequences=True, dropout=rnn_drop_out, recurrent_dropout=rnn_state_drop_out),

                name='bi_lstm_layer' + str(i))(x)

            x = layers.BatchNormalization(name='rnn_batch_norm_layer' + str(i))(x)



    for i in range(len(filter_sizes)):

        if is_integer(dilation_rates):

            di_rate = dilation_rates

        else:

            di_rate = dilation_rates[i]

        

        if is_integer(strides):

            std = strides

        else:

            std = strides[i]

            

        x = layers.Conv1D(num_filters[i], kernel_size=filter_sizes[i], padding=padding, activation='relu', dilation_rate=di_rate, strides=std)(x)

        

    if pooling == 'max':

        x = layers.GlobalMaxPooling1D(name='global_pooling_layer')(x)

    else:

        x = layers.GlobalAveragePooling1D(name='global_pooling_layer')(x)



    x = layers.BatchNormalization(name="batch_norm_layer")(x)

    x = layers.Dropout(cnn_drop_out, name='conv_dropout_layer')(x)



    # MLP Layers

    for i in range(mlp_depth - 1):

        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)

        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)



    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)



    model = models.Model(inputs, outputs)



    if return_customized_layers:

        return model, dict()



    return model
from keras.utils import model_to_dot

from keras import models

from keras import layers



import matplotlib.pyplot as plt

from IPython.display import SVG
histories = list()

iterations = list()

model_builders = list()
def build_model1():

    voca_dim = embedding_matrix.shape[0]

    time_steps = max_len

    output_dim = led.classes_.shape[0]

    mlp_dim = 50

    num_filters = [128, 128, 128]

    filter_sizes = [1, 3, 5]

    item_embedding = embedding_matrix

    mlp_depth = 2

    cnn_drop_out = 0.2

    mlp_drop_out = 0.2

    padding = 'causal'



    return build_cnn_model(

        voca_dim, time_steps, output_dim, mlp_dim, num_filters, filter_sizes, 

        item_embedding=item_embedding, mlp_depth=2, cnn_drop_out=cnn_drop_out,

        padding=padding,

        return_customized_layers=True

    )



model_builders.append(build_model1)
model, cnn_cl = build_model1()

print(model.summary())
adam = ko.Nadam()

model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy",])



file_path = "best_cnn_model.hdf5"

check_point = kc.ModelCheckpoint(file_path, monitor = "val_sparse_categorical_accuracy", verbose = 1, save_best_only = True, mode = "max")

early_stop = kc.EarlyStopping(monitor = "val_sparse_categorical_accuracy", mode = "max", patience=3)

history = model.fit(X_train, y_train, batch_size=500, epochs=20, validation_split=0.1, callbacks = [check_point, early_stop])



histories.append(np.max(np.asarray(history.history['val_sparse_categorical_accuracy'])))

iterations.append(np.argmax(np.asarray(history.history['val_sparse_categorical_accuracy'])))

del model, history

gc.collect()
def build_model2():

    voca_dim = embedding_matrix.shape[0]

    time_steps = max_len

    output_dim = led.classes_.shape[0]

    rnn_dim = 100

    mlp_dim = 50

    item_embedding = embedding_matrix

    rnn_depth=1

    mlp_depth = 2

    rnn_drop_out = 0.3

    rnn_state_drop_out = 0.3

    mlp_drop_out = 0.2

    num_att_channel = 1

    gpu=True

    

    return build_birnn_attention_model(

        voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, 

        item_embedding=item_embedding, rnn_depth=rnn_depth, mlp_depth=mlp_depth, num_att_channel=num_att_channel,

        rnn_drop_out=rnn_drop_out, rnn_state_drop_out=rnn_state_drop_out,

        gpu=gpu, return_customized_layers=True

    )



model_builders.append(build_model2)
model, rnn_cl = build_model2()

print(model.summary())
adam = ko.Nadam(clipnorm=2.0)

model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy",])



file_path = "best_birnn_attention_model.hdf5"

check_point = kc.ModelCheckpoint(file_path, monitor = "val_sparse_categorical_accuracy", verbose = 1, save_best_only = True, mode = "max")

early_stop = kc.EarlyStopping(monitor = "val_sparse_categorical_accuracy", mode = "max", patience=3)

history = model.fit(X_train, y_train, batch_size=500, epochs=20, validation_split=0.1, callbacks = [check_point, early_stop])



histories.append(np.max(np.asarray(history.history['val_sparse_categorical_accuracy'])))

iterations.append(np.argmax(np.asarray(history.history['val_sparse_categorical_accuracy'])))

del model, history

gc.collect()
def build_model3():

    voca_dim = embedding_matrix.shape[0]

    time_steps = max_len

    output_dim = led.classes_.shape[0]

    rnn_dim = 100

    mlp_dim = 50

    item_embedding = embedding_matrix

    rnn_depth=1

    mlp_depth = 2

    num_filters = [128, 128, 128]

    filter_sizes = [1, 3, 5]

    cnn_drop_out = 0.2

    rnn_drop_out = 0.3

    rnn_state_drop_out = 0.3

    mlp_drop_out = 0.2

    padding = 'causal'

    gpu=True

    

    return build_birnn_cnn_model(

        voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, num_filters, filter_sizes, 

        item_embedding=item_embedding, rnn_depth=rnn_depth, mlp_depth=mlp_depth,

        rnn_drop_out=rnn_drop_out, rnn_state_drop_out=rnn_state_drop_out, cnn_drop_out=cnn_drop_out,

        padding=padding,

        gpu=gpu, return_customized_layers=True

    )



model_builders.append(build_model3)
model, rc_cl = build_model3()

print(model.summary())
adam = ko.Nadam(clipnorm=2.0)

model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy",])



file_path = "best_birnn_cnn_model.hdf5"

check_point = kc.ModelCheckpoint(file_path, monitor = "val_sparse_categorical_accuracy", verbose = 1, save_best_only = True, mode = "max")

early_stop = kc.EarlyStopping(monitor = "val_sparse_categorical_accuracy", mode = "max", patience=3)

history = model.fit(X_train, y_train, batch_size=500, epochs=20, validation_split=0.1, callbacks = [check_point, early_stop])



histories.append(np.max(np.asarray(history.history['val_sparse_categorical_accuracy'])))

iterations.append(np.argmax(np.asarray(history.history['val_sparse_categorical_accuracy'])))

del model, history

gc.collect()
def build_model4():

    voca_dim = embedding_matrix.shape[0]

    time_steps = max_len

    output_dim = led.classes_.shape[0]

    rnn_dim = 100

    mlp_dim = 50

    item_embedding = embedding_matrix

    rnn_depth=1

    mlp_depth = 2

    num_filters = [128, 256, 512]

    filter_sizes = [1, 3, 5]

    dilation_rates = [1, 2, 4]

    strides=1

    cnn_drop_out = 0.2

    rnn_drop_out = 0.3

    rnn_state_drop_out = 0.3

    mlp_drop_out = 0.2

    padding = 'causal'

    gpu=True

    

    return build_birnn_hierarchy_cnn_model(

        voca_dim, time_steps, output_dim, rnn_dim, mlp_dim, num_filters, filter_sizes, 

        dilation_rates=dilation_rates, strides=strides,

        item_embedding=item_embedding, rnn_depth=rnn_depth, mlp_depth=mlp_depth,

        rnn_drop_out=rnn_drop_out, rnn_state_drop_out=rnn_state_drop_out, cnn_drop_out=cnn_drop_out,

        padding=padding,

        gpu=gpu, return_customized_layers=True

    )



model_builders.append(build_model4)
model, rhc_cl = build_model4()

print(model.summary())
adam = ko.Nadam(clipnorm=2.0)

model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy",])



file_path = "best_birnn_hierarchy_cnn_model.hdf5"

check_point = kc.ModelCheckpoint(file_path, monitor = "val_sparse_categorical_accuracy", verbose = 1, save_best_only = True, mode = "max")

early_stop = kc.EarlyStopping(monitor = "val_sparse_categorical_accuracy", mode = "max", patience=3)

history = model.fit(X_train, y_train, batch_size=500, epochs=20, validation_split=0.1, callbacks = [check_point, early_stop])



histories.append(np.max(np.asarray(history.history['val_sparse_categorical_accuracy'])))

iterations.append(np.argmax(np.asarray(history.history['val_sparse_categorical_accuracy'])))

del model, history

gc.collect()
histories = np.asarray(histories)



model_paths = [

    "best_cnn_model.hdf5",

    "best_birnn_attention_model.hdf5",

    "best_birnn_cnn_model.hdf5",

    "best_birnn_hierarchy_cnn_model.hdf5"

]



cls =[

    cnn_cl, rnn_cl, rc_cl, rhc_cl

]



pred = list()

for idx in range(len(model_paths)):

    model = models.load_model(model_paths[idx], cls[idx])

    pred_tmp = model.predict(X_test, batch_size = 1024, verbose = 1)

    pred.append(np.round(np.argmax(pred_tmp, axis=1)).astype(int))
def majority_vote(preds_data_point):

    unique, counts = np.unique(preds_data_point, return_counts=True)

    idx = np.argmax(counts)

    return unique[idx]



pred = np.asarray(pred)

predictions = list()

for i in range(pred.shape[1]):

    predictions.append(majority_vote(pred[:, i]))

predictions = np.asarray(predictions)



test_not_overlap_df = test_df[~overlap_boolean_mask_test]

test_not_overlap_df['Sentiment'] = predictions



res_df = pd.concat([overlapped, test_not_overlap_df], sort=True)[sub_df.columns.values.tolist()]



assert sub_df.shape[0] == res_df.shape[0]

assert sub_df.shape[1] == res_df.shape[1]



res_df.to_csv("submission.csv", index=False)