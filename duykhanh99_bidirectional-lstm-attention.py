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
from tqdm import tqdm

tqdm.pandas()
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', index_col='id')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')
train.head()
test.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.title("Distribution of target in the train set")

sns.distplot(train['target'],kde=True,hist=False, bins=120, label='target')

plt.legend(); plt.show()
def plot_features_distribution(features, title):

    plt.figure(figsize=(12,6))

    plt.title(title)

    for feature in features:

        sns.distplot(train[feature],kde=True,hist=False, bins=120, label=feature)

    plt.xlabel('')

    plt.legend()

    plt.show()
features = ['severe_toxicity', 'obscene','identity_attack','insult','threat']

plot_features_distribution(features, "Distribution of additional toxicity features in the train set")
features = ['asian', 'black', 'jewish', 'latino', 'other_race_or_ethnicity', 'white']

plot_features_distribution(features, "Distribution of race and ethnicity features values in the train set")
features = ['female', 'male', 'transgender', 'other_gender']

plot_features_distribution(features, "Distribution of gender features values in the train set")
features = ['bisexual', 'heterosexual', 'homosexual_gay_or_lesbian', 'other_sexual_orientation']

plot_features_distribution(features, "Distribution of sexual orientation features values in the train set")
features = ['atheist','buddhist',  'christian', 'hindu', 'muslim', 'other_religion']

plot_features_distribution(features, "Distribution of religion features values in the train set")
features = ['intellectual_or_learning_disability', 'other_disability', 'physical_disability', 'psychiatric_or_mental_illness']

plot_features_distribution(features, "Distribution of disability features values in the train set")
def plot_count(feature, title,size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(train))

    g = sns.countplot(train[feature], order = train[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()   
plot_count('rating','rating')
plot_count('funny','funny votes given',3)
plot_count('wow','wow votes given',3)
plot_count('sad','sad votes given',3)
plot_count('likes','likes given',3)
plot_count('disagree','disagree given',3)
features = ['sexual_explicit']

plot_features_distribution(features, "Distribution of sexual explicit values in the train set")
from wordcloud import WordCloud, STOPWORDS

import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import nltk

from gensim import corpora, models

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(train['comment_text'].sample(20000), title = 'Prevalent words in comments - train data')
from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers



class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim
from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler

from keras.losses import binary_crossentropy

from keras import backend as K
EMBEDDING_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]

NUM_MODELS = 2

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

MAX_LEN = 220
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)





def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix
def custom_loss(y_true, y_pred):

    return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]
def build_model(embedding_matrix, num_aux_targets, loss_weight):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([

        Attention(MAX_LEN)(x),

        GlobalMaxPooling1D()(x),

        #GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')



    return model
def preprocess(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



x_train = preprocess(train['comment_text'])
identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

# Overall

weights = np.ones((len(x_train),)) / 4

# Subgroup

weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4

# Background Positive, Subgroup Negative

weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +

   (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

# Background Negative, Subgroup Positive

weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +

   (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

loss_weight = 1.0 / weights.mean()
y_train = np.vstack([(train['target'].values>=0.5).astype(np.int),weights]).T

y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values

x_test = preprocess(test['comment_text'])
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(x_train) + list(x_test))



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
import pickle

import gc



with open('temporary.pickle', mode='wb') as f:

    pickle.dump(x_test, f) # use temporary file to reduce memory



del identity_columns, weights, tokenizer, train, test, x_test

gc.collect()



checkpoint_predictions = []

weights = []
build_model(embedding_matrix, y_aux_train.shape[-1], loss_weight).summary()
y_aux_train.shape[-1]
for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weight)

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=1,

            callbacks=[

                LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))

            ]

        )

        with open('temporary.pickle', mode='rb') as f:

            x_test = pickle.load(f) # use temporary file to reduce memory

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

        del x_test

        gc.collect()

        weights.append(2 ** global_epoch)

    del model

    gc.collect()
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
df_submit = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

df_submit.prediction = predictions

df_submit.to_csv('submission.csv', index=False)