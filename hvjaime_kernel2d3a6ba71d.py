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

import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler
NUM_MODELS = 2

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

MAX_LEN = 220

IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
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
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",

                       "could've": "could have", "couldn't": "could not", "didn't": "did not", 

                       "doesn't": "does not", "don't": "do not", "hadn't": "had not", 

                       "hasn't": "has not", "haven't": "have not", "he'd": "he would",

                       "he'll": "he will", "he's": "he is", "how'd": "how did", 

                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will",

                       "I'll've": "I will have","I'm": "I am", "I've": "I have",

                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will",

                       "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not",

                       "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 

                       "it'll've": "it will have","it's": "it is", "let's": "let us",

                       "ma'am": "madam", "mayn't": "may not", "might've": "might have",

                       "mightn't": "might not","mightn't've": "might not have", 

                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 

                       "needn't": "need not", "needn't've": "need not have",

                       "o'clock": "of the clock", "oughtn't": "ought not", 

                       "oughtn't've": "ought not have", "shan't": "shall not",

                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 

                       "she'd've": "she would have", "she'll": "she will",

                       "she'll've": "she will have", "she's": "she is", "should've": "should have", 

                       "shouldn't": "should not", "shouldn't've": "should not have", 

                       "so've": "so have","so's": "so as", "this's": "this is",

                       "that'd": "that would", "that'd've": "that would have", "that's": "that is",

                       "there'd": "there would", "there'd've": "there would have", 

                       "there's": "there is", "here's": "here is","they'd": "they would", 

                       "they'd've": "they would have", "they'll": "they will", 

                       "they'll've": "they will have", "they're": "they are", 

                       "they've": "they have", "to've": "to have", "wasn't": "was not",

                       "we'd": "we would", "we'd've": "we would have", "we'll": 

                       "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",

                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have",

                       "what're": "what are",  "what's": "what is", "what've": "what have", 

                       "when's": "when is", "when've": "when have", "where'd": "where did", 

                       "where's": "where is", "where've": "where have", "who'll": "who will", 

                       "who'll've": "who will have", "who's": "who is", "who've": "who have", 

                       "why's": "why is", "why've": "why have", "will've": "will have",

                       "won't": "will not", "won't've": "will not have", "would've": "would have",

                       "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 

                       "y'all'd": "you all would","y'all'd've": "you all would have",

                       "y'all're": "you all are","y'all've": "you all have","you'd": "you would", 

                       "you'd've": "you would have", "you'll": "you will", 

                       "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
train_df['comment_text'] = train_df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
test_df['comment_text'] = test_df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
x_train = train_df[TEXT_COLUMN].astype(str)

y_train = train_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str)
x_train.shape, y_train.shape, x_test.shape
for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train_df[column] = np.where(train_df[column] >= 0.5, True, False)
tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(list(x_train) + list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
#sample_weights = np.ones(len(x_train), dtype=np.float32)

#sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)

#sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)

#sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5

#sample_weights /= sample_weights.mean()
EMBEDDING_FILES = [

    '../input/fast-track-300d-2m-vec/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]
embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
checkpoint_predictions = []

weights = []
for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix, y_aux_train.shape[-1])

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=2,

            #sample_weight=[sample_weights.values, np.ones_like(sample_weights)],

            callbacks=[

                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))

            ]

        )

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

        weights.append(2 ** global_epoch)
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    'prediction': predictions

})

submission.to_csv('submission.csv', index=False)