# Math and linear algebra

import numpy as np



# Data manipulation

import pandas as pd



# Plots

import seaborn as sns

import matplotlib.style as style

import matplotlib.pyplot as plt



# Keras: high-level neural network API 

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout

from keras.preprocessing import text, sequence



# Sk-learn

from sklearn.feature_extraction.text import CountVectorizer



# Gensim: NLP library. We will use this to load the word embedding vectors

from gensim.models import KeyedVectors



import re



# Keras: deep learning API

import keras

print(keras.__version__)

import tensorflow

print(tensorflow.__version__)



# Embedding files

EMBEDDING_FILES = [

    '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim',

    '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'

    #'../input/gensim-embeddings-dataset/GoogleNews-vectors-negative300.gensim'

]



# 

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 20

MAX_LEN = 220

TEXT_COLUMN = 'text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
print("Read Data")

train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

train_df.head(5)
print("There are", train_df.shape[0], "tweets in the training dataset.")
style.use("ggplot")

x = train_df.isna().sum()*100.0/train_df.shape[0]

ax = sns.barplot(x.index, x, palette="Blues_d")

ax.set_ylabel("%")
v = train_df["target"].value_counts()

sns.barplot(v.index, v)
fig,axes=plt.subplots(1,2,figsize=(10,5), sharey='row')

fig.suptitle("Tweet length in characters")

for i in range(2):

    sns.distplot(train_df[train_df["target"] == i]["text"].str.len(), ax=axes[i], color=sns.color_palette()[i])

    axes[i].set_xlabel("Real = "+str(i))
fig,axes=plt.subplots(1,2,figsize=(10,5), sharey='row')

fig.suptitle("Tweet length in words")

for i in range(2):

    sns.distplot(train_df[train_df["target"] == i]["text"].str.split().map(lambda x: len(x)), ax=axes[i], color=sns.color_palette()[i])

    axes[i].set_xlabel("Real = "+str(i))
fig,axes=plt.subplots(1,2,figsize=(9,5))

fig.suptitle("NaNs by variable")

v = train_df["keyword"].isnull().groupby(train_df["target"]).sum().astype(int)

sns.barplot(v.index, v, ax=axes[0])

axes[0].set_ylabel("# of NaNs")

axes[0].set_xlabel("")

axes[0].set_title("Keyword")



v = train_df["location"].isnull().groupby(train_df["target"]).sum().astype(int)

sns.barplot(v.index, v, ax=axes[1])

axes[1].set_ylabel("# of NaNs")

axes[1].set_xlabel("")

axes[1].set_title("Location")



fig.tight_layout(rect=[0, 0.03, 1, 0.90])
# Most common words

from collections import Counter

fig, axes = plt.subplots(1, 2, figsize=(14,5))

fig.suptitle("Most common words")

for i in range(2):

    word_list = [x for sublist in list(train_df[train_df["target"] == i]["text"].map(lambda x: x.split()).values) for x in sublist]

    top = sorted(Counter(word_list).items(), key=lambda x:x[1], reverse=True)[:10]

    x, y = zip(*top)

    axes[i].bar(x, y)

    axes[i].set_xlabel("Real = "+str(i))
# Data processing

import string



def remove_urls(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



def process_tweets(tweets):

    tweets = remove_urls(tweets)

    tweets = remove_html(tweets)

    tweets = remove_punct(tweets)

    return tweets



train_df["text"] = train_df["text"].map(process_tweets)

test_df["text"] = test_df["text"].map(process_tweets)
x_train = train_df[TEXT_COLUMN].astype(str)

y_train = train_df[TARGET_COLUMN].values

x_test = test_df[TEXT_COLUMN].astype(str)



tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE, lower=False)

tokenizer.fit_on_texts(list(x_train) + list(x_test))



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)



checkpoint_predictions = []

weights = []



def build_matrix(word_index, path):

    embedding_index = KeyedVectors.load(path, mmap='r')

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        for candidate in [word, word.lower()]:

            if candidate in embedding_index:

                embedding_matrix[i] = embedding_index[candidate]

                break

    return embedding_matrix



embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)



def build_model(embedding_matrix):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.5)(x)

    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)

    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)

    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x)

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=result)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
print("Start Modeling")

model = build_model(embedding_matrix)

model.fit(

    x_train,

    y_train,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    validation_split=0.2,

    verbose=1

)

print("Modeling Complete")
predictions = model.predict(x_test)

submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    TARGET_COLUMN: np.round(predictions).astype(int).reshape(x_test.shape[0])

})

submission.to_csv('lstm_submission.csv', index=False)

submission.head()