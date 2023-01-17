import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, Dropout, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras import callbacks

from keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split





#crawl-300d-2M.vec--> https://fasttext.cc/docs/en/english-vectors.html

#When pre-train embedding is helpful? https://www.aclweb.org/anthology/N18-2084

#There are many pretrained word embedding models: 

#fasttext, GloVe, Word2Vec, etc

#crawl-300d-2M.vec is trained from Common Crawl (a website that collects almost everything)

#it has 2 million words. Each word is represent by a vector of 300 dimensions.



#https://nlp.stanford.edu/projects/glove/

#GloVe is similar to crawl-300d-2M.vec. Probably, they use different algorithms.

#glove.840B.300d.zip: Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)

#tokens mean words. It has 2.2M different words and 840B (likely duplicated) words in total



#note that these two pre-trained models give 300d vectors.

EMBEDDING_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]



NUM_MODELS = 2

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4 

MAX_LEN = 200

MAX_FEATURES = 120000

IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
#functions to build our embedding matrix

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
#importing the data

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



x_train = train_df[TEXT_COLUMN].astype(str)

y_aux_train = train_df[AUX_COLUMNS].values

y_train = train_df[TARGET_COLUMN].values

x_test = test_df[TEXT_COLUMN].astype(str)
#tokenizing the corpus, limiting the tokenizer to 120000 words

for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train_df[column] = np.where(train_df[column] >= 0.5, True, False)



tokenizer = text.Tokenizer(num_words=MAX_FEATURES, filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(list(x_train) + list(x_test))



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)
#deciding the value MAX_LEN

totalNumWords = [len(one_comment) for one_comment in x_train]

plt.hist(totalNumWords)

plt.show()
#making sure that every sentence is of equal length by adding padding

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
#building the embedding matrix

embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
#function to build the model

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

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    return model
checkpoint_predictions = []
#fitting model on whole training data

for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix, y_aux_train.shape[-1])

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=1,

            callbacks=[

                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))

            ]

        )

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
predictions = np.average(checkpoint_predictions, axis=0)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

submission['prediction'] = predictions

submission.to_csv('submission.csv', index=False)