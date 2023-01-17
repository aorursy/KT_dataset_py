import numpy as np

import pandas as pd

import tensorflow

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from tensorflow.keras.layers import LSTM, GlobalMaxPooling1D, GlobalAveragePooling1D

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.callbacks import LearningRateScheduler
EMBEDDING_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]
NUM_MODELS = 1

BATCH_SIZE = 512

LSTM_UNITS = 64

DENSE_HIDDEN_UNITS = 2 * LSTM_UNITS

EPOCHS = 1

MAX_LEN = 220
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv',low_memory=True,nrows = 5000) #Use full data here

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv',low_memory=True)
train.head()
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

x_train = preprocess(train['comment_text'])

#y_train = np.where(train['target'] >= 0.5, 1, 0)
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

x_test = preprocess(test['comment_text'])
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(x_train) + list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
import gc

del train

gc.collect()
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
embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
del tokenizer

gc.collect()
checkpoint_predictions = []

weights = []
def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words) #Finds word embeddings for each word

    x = SpatialDropout1D(0.3)(x) #This version performs the same function as Dropout, however it drops entire 1D feature maps instead of individual elements

    x = LSTM(LSTM_UNITS, return_sequences=True)(x)

    x = LSTM(LSTM_UNITS, return_sequences=True)(x)

    hidden = concatenate([

        GlobalMaxPooling1D()(x), 

        GlobalAveragePooling1D()(x),#layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input 

        #of variable length in the simplest way possible.

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS)(hidden)]) #This fixed-length output vector is piped through a fully-connected (Dense) layer with x hidden units.

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS)(hidden)])

    #result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=aux_result)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    

    return model
model = build_model(embedding_matrix, y_aux_train.shape[-1])
model.summary()
t_model = model.fit(

            x_train,

            y_aux_train,

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=1

        )
y_aux_train.columns
y_pred = model.predict(x_train[1].reshape(1,220))
y_pred