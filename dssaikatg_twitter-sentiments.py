# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install twython
import string

import re

import seaborn as sns

import matplotlib.pyplot as plt

import pickle



from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import f1_score, classification_report, confusion_matrix



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model

from keras.layers import Embedding, Dense, Dropout, LSTM, GRU, BatchNormalization, Activation, Input, Reshape

from keras.initializers import Constant

from keras import backend as K

import keras



import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer, WordNetLemmatizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import mark_negation

# nltk.download('stopwords')

# nltk.download('wordnet')

# nltk.download('averaged_perceptron_tagger')

# nltk.download('vader_lexicon')



import gensim



from tqdm import tqdm

tqdm.pandas(ncols=70)
df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1", names=["target", "ids", "date", "flag", "user", "text"])

print(df.shape)

df.head()
# Only Negative(0) and Positive(4) labels are present.

df.target.unique()
df['target'] = df['target'].apply(lambda x: x if x==0 else 1)

df.target.unique()
# Negative-0 and Positive-1

df.target.value_counts().plot(kind='bar');
[print(i) for i in df.iloc[:5]['text']];
# Dataset doesn't contain any bitcoin hashtags

df[df.text.str.contains('#bitcoin')]
stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")

sia = SentimentIntensityAnalyzer()



def preprocess(tweet):

#     wnl = WordNetLemmatizer()

    cleaned_tweet = []



    words = tweet.split()

    for word in words:

        # Skip Hyperlinks and Twitter Handles @<user>

        if ('http' in word) or ('.com' in word) or ('www.' in word) or (word.startswith('@')):

            continue



        # Remove Digits and Special Characters

        temp = re.sub(f'[^{string.ascii_lowercase}]', '', word.lower())



        # Remove words with less than 3 characters

        if len(temp) < 3:

            continue



        # Store the Stemmed version of the word

        temp = stemmer.stem(temp)



        if len(temp) > 0:

            cleaned_tweet.append(temp)



    return ' '.join(cleaned_tweet)
df['cleaned_tweet'] =  df['text'].progress_apply(preprocess)
# Featurizations

df['nltkdict'] = df['text'].progress_apply(lambda x: sia.polarity_scores(x))

df['nltk_compound'] = df['nltkdict'].progress_apply(lambda x: x['compound'])

df['nltk_neg'] = df['nltkdict'].progress_apply(lambda x: x['neg'])

df['nltk_pos'] = df['nltkdict'].progress_apply(lambda x: x['pos'])

df['nltk_neu'] = df['nltkdict'].progress_apply(lambda x: x['neu'])



df['count_negations'] = df['text'].progress_apply(lambda x: len([w for w in mark_negation(x.split()) if '_NEG' in w]))

df['count_!'] = df['text'].progress_apply(lambda x: x.count('!'))

df['count_#'] = df['text'].progress_apply(lambda x: x.count('#'))

df['count_@'] = df['text'].progress_apply(lambda x: x.count('@'))



df['count_upper'] = df['text'].progress_apply(lambda x: len([w for w in x.split() if w.isupper()]))

df['count_words'] = df['text'].progress_apply(lambda x: len(x.split()))

df['tweet_length'] = df['text'].progress_apply(lambda x: len(x))
df.head()
data = df[['target', 'cleaned_tweet']].copy()

data.head()
X = data['cleaned_tweet']

y = data['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=data['target'], train_size=0.7, random_state=42)

X_train.shape, X_test.shape
# Create a Tokenizer object and fit it on the entire dataset (Train + Test)

# This will contain the vocab space of the dataset

tokenizer = Tokenizer()

total_tweets = data.cleaned_tweet

tokenizer.fit_on_texts(total_tweets)



# Calculate Max Seq Length to pad sequences

max_seq_len = max([len(s.split()) for s in total_tweets])



# Define Vocab size

vocab_size = len(tokenizer.word_index) + 1



# Create the sequence of Indices from Text

X_train_tokens = tokenizer.texts_to_sequences(X_train)

X_test_tokens = tokenizer.texts_to_sequences(X_test)



# Pad the sequences to make the length equal to max_seq_len

X_train_pad = pad_sequences(X_train_tokens, maxlen=max_seq_len, padding='post')

X_test_pad = pad_sequences(X_test_tokens, maxlen=max_seq_len, padding='post')
# dir(tokenizer)
# Embedding Layer --> Turns positive integers (indexes) into dense vectors of fixed size.

EMBEDDING_DIM = 300



model = Sequential()

model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_seq_len))

model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='relu'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(X_train_pad, y_train, batch_size=128, epochs=5, validation_split=0.3, verbose=1)
lstm_preds = model.predict(X_test_pad)

lstm_predictions = lstm_preds > 0.5



print(confusion_matrix(y_test, lstm_predictions))

print()

print(classification_report(y_test, lstm_predictions))
plt.figure(figsize=(16, 8))

plt.plot(range(1, 6), history.history['val_accuracy'], 'b', label='val_acc')

plt.plot(range(1, 6), history.history['accuracy'], 'b--', label='train_acc')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
EMBEDDING_DIM = 100

tweets = data.cleaned_tweet.progress_apply(lambda x: x.split())



# Train Word2Vec Model

model_w2v = gensim.models.Word2Vec(sentences=tweets, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)



# Vocab list

words = list(model_w2v.wv.vocab)
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():

    if i > vocab_size:

        continue

    try:

        embedding_matrix[i] = model_w2v.wv[word]

    except:

        embedding_matrix[i] = np.zeros(EMBEDDING_DIM)
# Embedding Layer --> Turns positive integers (indexes) into dense vectors of fixed size.

K.clear_session()

EMBEDDING_DIM = 100



model = Sequential()

model.add(Embedding(vocab_size, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), input_length=max_seq_len, trainable=False))

model.add(LSTM(units=32, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))

model.add(LSTM(units=16, dropout=0.4, recurrent_dropout=0.4))

model.add(Dense(1, activation='relu'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(X_train_pad, y_train, batch_size=128, epochs=10, validation_split=0.3, verbose=1)
print("maxlen =", max_seq_len)

lstm_preds = model.predict(X_test_pad)

lstm_predictions = lstm_preds > 0.5



print(confusion_matrix(y_test, lstm_predictions))

print()

print(classification_report(y_test, lstm_predictions))
# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
# pickle.dump(model, open("model.p", "wb"))

# pickle.dump(tokenizer, open("tokenizer.p", "wb"))

os.chdir(r'/kaggle/working')
from IPython.display import FileLink

# FileLink(r'model.p')

# FileLink(r'tokenizer.p')

FileLink(r'model.json')

# FileLink(r'model.h5')
# Embedding Layer --> Turns positive integers (indexes) into dense vectors of fixed size.

K.clear_session()

EMBEDDING_DIM = 300



model = Sequential()

model.add(Embedding(vocab_size, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), input_length=max_seq_len, trainable=True))

model.add(LSTM(units=32, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))

model.add(LSTM(units=16, dropout=0.4, recurrent_dropout=0.4))

model.add(Dense(1, activation='relu'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(X_train_pad, y_train, batch_size=128, epochs=5, validation_split=0.3, verbose=1)
lstm_preds = model.predict(X_test_pad)

lstm_predictions = lstm_preds > 0.5



print(confusion_matrix(y_test, lstm_predictions))

print()

print(classification_report(y_test, lstm_predictions))
df.head(2)
def model_lstm_w2v(df, embedding_matrix, EMBEDDING_DIM=300, trainable=False):

    train_len = int(df.shape[0]*0.7)

    all_tweets = df.cleaned_tweet.apply(lambda x: x.split())

    

    X_train = df['cleaned_tweet'][:train_len]

    y_train = df['target'][:train_len]

    X_test = df['cleaned_tweet'][train_len:]

    

    X_feats_train = df.drop(['ids', 'date', 'flag', 'user', 'target', 'text', 'cleaned_tweet', 'nltkdict'], axis=1)[:train_len]

    X_feats_test = df.drop(['ids', 'date', 'flag', 'user', 'target', 'text', 'cleaned_tweet', 'nltkdict'], axis=1)[train_len:]

    

    # Create a Tokenizer object and fit it on the entire dataset (Train + Test)

    # This will contain the vocab space of the dataset

    tokenizer = Tokenizer()

    total_tweets = df.cleaned_tweet

    tokenizer.fit_on_texts(total_tweets)



    # Calculate Max Seq Length to pad sequences

    max_seq_len = max([len(s.split()) for s in total_tweets])



    # Define Vocab size

    vocab_size = len(tokenizer.word_index) + 1

    

    # Create the sequence of Indices from Text

    X_train_tokens = tokenizer.texts_to_sequences(X_train)

    X_test_tokens = tokenizer.texts_to_sequences(X_test)



    # Pad the sequences to make the length equal to max_seq_len

    X_train_pad = pad_sequences(X_train_tokens, maxlen=max_seq_len, padding='post')

    X_test_pad = pad_sequences(X_test_tokens, maxlen=max_seq_len, padding='post')

    

#     # Train Word2Vec Model on the entire Tweet dataset - Train + Test

#     model_w2v = gensim.models.Word2Vec(sentences=all_tweets, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)



#     # Vocab list

#     words = list(model_w2v.wv.vocab)



#     # Build Embedding Matrix

#     embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

#     for word, i in tokenizer.word_index.items():

#         if i > vocab_size:

#             continue

#         try:

#             embedding_matrix[i] = model_w2v.wv[word]

#         except:

#             embedding_matrix[i] = np.zeros(EMBEDDING_DIM)



    # Embedding Layer --> Turns positive integers (indexes) into dense vectors of fixed size.    

    print(embedding_matrix.shape, X_feats_train.shape)

    

    def get_model(trainable):

        inp_1 = Input(shape=(max_seq_len, ))

        x = Embedding(vocab_size, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), trainable=trainable)(inp_1)



        x = LSTM(units=32, dropout=0.4, recurrent_dropout=0.4, return_sequences=True)(x)

        x = LSTM(units=16, dropout=0.4, recurrent_dropout=0.4)(x)



        inp_2 = Input(shape=(X_feats_train.shape[1],))

        x = keras.layers.concatenate([x, inp_2], axis=-1)



        x = Dense(100, kernel_initializer='he_normal')(x)

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = Dropout(0.5)(x)

        x = Dense(10, activation='relu', kernel_initializer='he_normal')(x)



        out = Dense(1, activation='relu')(x)



        model = Model(inputs=[inp_1, inp_2], outputs=out)



        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        

        return model



    model = get_model(trainable)

    history = model.fit([X_train_pad, X_feats_train], y_train, batch_size=128, epochs=5, validation_split=0.3, verbose=1)



    # Predict on the Train dataset

    train_w2v_preds = model.predict([X_train_pad, X_feats_train])

    train_w2v_predictions = train_w2v_preds > 0.5

    

    # Predict on the Test dataset

    w2v_preds = model.predict([X_test_pad, X_feats_test])

    w2v_predictions = w2v_preds > 0.5



    return (train_w2v_preds, train_w2v_predictions, w2v_preds, w2v_predictions)
(train_preds, train_predictions, lstm_preds, lstm_predictions) = model_lstm_w2v(df, embedding_matrix)
print(confusion_matrix(y_test, lstm_predictions))

print()

print(classification_report(y_test, lstm_predictions))