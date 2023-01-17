import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler



import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px

# import plotly.plotly as py

import plotly.offline as pyo

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



import os

print(os.listdir("../input"))
NUM_MODELS = 2

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

MAX_LEN = 220
print(os.listdir("../input"))

print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

sample_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
train_df.head(2)

test_df.head(2)
sample_df.head(2)
IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'
x_train = train_df[TEXT_COLUMN].astype(str)

y_train = train_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str)



for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train_df[column] = np.where(train_df[column] >= 0.5, True, False)
x_train.head()
y_train[:5]
train_df.head(5)
from keras.preprocessing import text

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'



tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(list(x_train) + list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)
print(x_train[0])
from keras.preprocessing import sequence

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

print(x_train[0])
sample_weights = np.ones(len(x_train), dtype=np.float32)
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)



"""

For reminder on identity columns:

IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

"""
data = [go.Histogram(x=sample_weights)]

layout = {'title': 'Distribution of weights after adding identity_columns'}

iplot({'data':data, 'layout':layout})
sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)

sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5

sample_weights /= sample_weights.mean()
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
path = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

with open(path) as f:

    i = 0

    for line in f:

        output1 = line.strip().split(' ')

        print(output1)

        print(type(output1))

        print('=====')

        i += 1

        if i == 5:

            break              
EMBEDDING_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]

embedding_file1 = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

embedding_file2 = '../input/glove840b300dtxt/glove.840B.300d.txt'
# embedding_index = load_embeddings(path)

# embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))

# for word, i in tokenizer.word_index.items():

#     try:

#         embedding_matrix[i] = embedding_index[word]

#     except KeyError:

#         pass  
print(embedding_matrix[:2])
embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
"""

Reminder: y_aux_train = train_df[AUX_COLUMNS].values

"""

num_aux_targets = y_aux_train.shape[-1]

print(num_aux_targets)


LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4
def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model
model = build_model(embedding_matrix, y_aux_train.shape[-1])
model.summary()
# checkpoint_predictions = []

# weights = []

# EPOCHS = 4

# NUM_MODELS = 2

# BATCH_SIZE = 512



# for model_idx in range(NUM_MODELS):  # Not sure why we use this, since model_idx is never used below

#     model = build_model(embedding_matrix, y_aux_train.shape[-1])

#     for global_epoch in range(EPOCHS):

#         model.fit(

#             x_train,

#             [y_train, y_aux_train],

#             batch_size=BATCH_SIZE,

#             epochs=1,

#             verbose=2,

#             sample_weight=[sample_weights.values, np.ones_like(sample_weights)],

#             callbacks=[

#                 LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))

#             ]

#         )

#         checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

#         weights.append(2 ** global_epoch)
# predictions = np.average(checkpoint_predictions, weights=weights, axis=0)



# submission = pd.DataFrame.from_dict({

#     'id': test_df.id,

#     'prediction': predictions

# })

# submission.to_csv('submission.csv', index=False)