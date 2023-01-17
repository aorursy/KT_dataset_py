import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import pickle

from tqdm import tqdm

from wordcloud import WordCloud





from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn import metrics

from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix,classification_report

%matplotlib inline

warnings.filterwarnings("ignore")

from IPython.display import Image,YouTubeVideo,HTML



#KERAS Import

from keras.models import Sequential, Model

from keras.utils import to_categorical,plot_model

from keras.layers import Dense, Activation

from keras.layers.normalization import BatchNormalization

from keras.initializers import he_normal,glorot_normal

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.layers import Dropout

from keras.layers import Embedding,CuDNNLSTM,CuDNNGRU, Flatten, Input, concatenate, Conv1D, GlobalMaxPooling1D, SpatialDropout1D, GlobalAveragePooling1D, Bidirectional

from keras.regularizers import l2

from keras.optimizers import Adam

from keras.initializers import Orthogonal

from keras.preprocessing.text import one_hot

from keras.constraints import max_norm

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers
df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

df.head()
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
print(df.iloc[28]['comment_text'])

print("Toxicity Level: ",df.iloc[28]['target'])
print(df.iloc[4]['comment_text'])

print("Toxicity Level: ",df.iloc[4]['target'])
# https://stackoverflow.com/a/47091490/4084039

import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
# https://gist.github.com/sebleier/554280

# we are removing the words from the stop words list: 'no', 'nor', 'not'

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"]
# Combining all the above statemennts 

preprocessed_comments = []

# tqdm is for printing the status bar

for sentence in tqdm(df['comment_text'].values):

    sent = decontracted(sentence)

    sent = sent.replace('\\r', ' ')

    sent = sent.replace('\\"', ' ')

    sent = sent.replace('\\n', ' ')

    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

    # https://gist.github.com/sebleier/554280

    sent = ' '.join(e for e in sent.split())

    preprocessed_comments.append(sent.lower().strip())
df['comment_text'] = preprocessed_comments
df['comment_text'][1]
# Combining all the above statemennts 

preprocessed_comments_test = []

# tqdm is for printing the status bar

for sentence in tqdm(test_df['comment_text'].values):

    sent = decontracted(sentence)

    sent = sent.replace('\\r', ' ')

    sent = sent.replace('\\"', ' ')

    sent = sent.replace('\\n', ' ')

    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

    # https://gist.github.com/sebleier/554280

    sent = ' '.join(e for e in sent.split())

    preprocessed_comments_test.append(sent.lower().strip())
test_df['comment_text'] = preprocessed_comments_test
train_len = len(df.index)
miss_val_train_df = df.isnull().sum(axis=0) / train_len

miss_val_train_df = miss_val_train_df[miss_val_train_df > 0] * 100

miss_val_train_df
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
for column in identity_columns + ['target']:

    df[column] = np.where(df[column] >= 0.5, True, False)
# Target variable as well

y = df['target'].values
train_df = df
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
MAX_VOCAB_SIZE = 100000

TOXICITY_COLUMN = 'target'

TEXT_COLUMN = 'comment_text'

MAX_SEQUENCE_LENGTH = 300



# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

tokenizer.fit_on_texts(train_df[TEXT_COLUMN])



# All comments must be truncated or padded to be the same length.

def padding_text(texts, tokenizer):

    return sequence.pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)
train_text = padding_text(train_df[TEXT_COLUMN], tokenizer)

train_y = to_categorical(train_df[TOXICITY_COLUMN])
# for submission purpose

test_text = padding_text(test_df[TEXT_COLUMN], tokenizer)
NUM_EPOCHS = 6

BATCH_SIZE = 1024
embeddings_index = {}

with open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec' ,encoding='utf8') as f:

  for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs
len(tokenizer.word_index)
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,300))

num_words_in_embedding = 0

for word, i in tokenizer.word_index.items():

  embedding_vector = embeddings_index.get(word)

  if embedding_vector is not None:

    num_words_in_embedding += 1

    # words not found in embedding index will be all-zeros.

    embedding_matrix[i] = embedding_vector
embedding_matrix.shape
input_text_bgru = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')

embedding_layer_bgru = Embedding(len(tokenizer.word_index) + 1,

                                    300,

                                    weights=[embedding_matrix],

                                    input_length=MAX_SEQUENCE_LENGTH,

                                    trainable=False)

g = embedding_layer_bgru(input_text_bgru)

g = SpatialDropout1D(0.4)(g)

g = Bidirectional(CuDNNGRU(64, return_sequences=True))(g)

att = Attention(MAX_SEQUENCE_LENGTH)(g)

g = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(g)

avg_pool = GlobalAveragePooling1D()(g)

max_pool = GlobalMaxPooling1D()(g)

g = concatenate([att, avg_pool, max_pool])

g = Dense(128, activation='relu')(g)

bgru_output = Dense(2, activation='softmax')(g)
model = Model(inputs=[input_text_bgru], outputs=[bgru_output])
plot_model(model, show_shapes=True, to_file='singlegru.png')

Image(filename="singlegru.png")
model.compile(loss='categorical_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])
SGRU_Model = model.fit(train_text,train_y,

              batch_size=BATCH_SIZE,

              epochs=NUM_EPOCHS)
predictions = model.predict(test_text)[:, 1]
df_submit = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

df_submit.prediction = predictions

df_submit.to_csv('submission.csv', index=False)