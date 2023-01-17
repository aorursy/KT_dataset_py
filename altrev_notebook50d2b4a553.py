import numpy as np

import pandas as pd

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import roc_auc_score

from datetime import date

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Embedding

from keras.layers import LSTM, SimpleRNN, GRU

from keras.callbacks import Callback, EarlyStopping



# read data

data = pd.read_csv('../input/Combined_News_DJIA.csv')

y = data.pop('Label').as_matrix()



# Use RenMai's fancy joining code

X_raw = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)
# Let's remove other characters, stop words and lemmatize and turn into sequences of numbers

# You might need this, I had them installed already:

# nltk.download(['stopwords', 'wordnet'])

stopwords = nltk.corpus.stopwords.words('english')

lemmatize = nltk.WordNetLemmatizer().lemmatize

tokenize = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize

alphabet = '''abcdefghijklmnopqrstuvwxyz0123456789 '''

# A dict for transforming into sequences:

char_to_int = dict((c, i) for i, c in enumerate(alphabet))

vocab = {}

count = 0

def normalize(o):

    global count

    r = []

    for t in tokenize(o):

        if t not in stopwords and len(t) > 2:

            t = lemmatize(t).lower()

            t= ''.join(lc for lc in t if lc in alphabet)

            if t in vocab:

                n = vocab[t]

            else:

                vocab[t] = count

                n = count

                count += 1

            r.append(n)

    return r

print('Normalize words and transform to sequences...')

X_text = [normalize(x) for x in X_raw]



# This vocabulary size might be a bit large, perhaps one should restrict it to the highest frequency scores

vocab_size = len(vocab)

print('Vocabulary size:' + str(vocab_size))
# Suggested train/test split. Kudos to RenMai, too

TRAINING_END = date(2014,12,31)

num_training = len(data[pd.to_datetime(data["Date"]) <= TRAINING_END])

X_train = X_text[:num_training]

X_test = X_text[num_training:]

y_train = y[:num_training]

y_test = y[num_training:]



# Pad sequences to maximum length.

X_train = sequence.pad_sequences(X_train)

X_test = sequence.pad_sequences(X_test)



print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)
# Build model

# High dropout, because this quickly overfits.

model = Sequential()

model.add(Embedding(vocab_size, 128, dropout=0.75))

model.add(LSTM(128, dropout_W=0.3, dropout_U=0.3))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Train...')

model.fit(X_train, y_train, batch_size=32, nb_epoch=11, validation_data=(X_test, y_test), verbose=0)
# predict and evaluate predictions

predictions = model.predict_proba(X_test)

print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions)))