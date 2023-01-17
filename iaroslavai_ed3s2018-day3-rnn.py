import pandas as pd
import numpy as np

lumber = pd.read_csv('../input/quandl-lumber-price-history/lumber.csv')
lumber = lumber[::-1]
lumber = lumber.drop('Date', axis=1)
display(lumber[:12])
from skimage.util import view_as_windows

history = view_as_windows(lumber.values, (10,7)).squeeze()
X = history[:, :-1, :]
y = history[:, -1, 0] > history[:, -2, 3]

print(X.shape)
print(y.shape)

print(X[0])
print(y[0])
# GRU, SimpleRNN, LSTM are recurrent layers
from keras.layers import Input, GRU, SimpleRNN, LSTM, Dense, LeakyReLU, Softmax
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.losses import sparse_categorical_crossentropy

#np.random.seed(1)
X_train, X_test, y_train, y_test = X[:1500], X[1500:], y[:1500], y[1500:]

m, s = np.mean(X_train, axis=(0, 1)), np.std(X_train, axis=(0, 1))
X_train = (X_train - m) / s
X_test = (X_test - m) / s

inp = Input(shape=(9, 7))
h = inp
# Task: try SimpleRNN, GRU, LSTM
h = GRU(16)(h)  # this takes as input a sequence, and returns last activation of RNN
h = Dense(2)(h)
h = Softmax()(h)

model = Model(inputs=[inp], outputs=[h])
model.compile(Adam(0.001), sparse_categorical_crossentropy, ['accuracy'])
model.fit(X_train, y_train, epochs=5, verbose=1)

# get the model performance
loss, acc = model.evaluate(X_test, y_test)
print(acc)
# Task: something missing here?
x = np.array([
    [
        [339., 341., 337.5, 339.3, 339.3, 159., 838.],
        [338.3, 338.5, 336., 336.5, 336.5, 149., 714.],
        [337.3, 337.3, 334., 335., 335., 278., 567.],
        [336., 336., 328.2, 329.1, 329.1, 326., 464.],
        [332., 336., 331., 331.2, 331.2, 167., 387.],
        [331.5, 332.4, 330.2, 330.4, 330.4, 97., 302.],
        [331.3, 334.8, 330., 331.5, 331.5, 243., 209.],
        [332.2, 334.5, 326.6, 326.6, 326.6, 123., 104.],
        [332.9, 332.9, 325., 325., 325., 83., 93.]
    ]
])
model.predict(x)
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

Xf = np.reshape(X, (len(X), -1))
X_train, X_test, y_train, y_test = Xf[:1500], Xf[1500:], y[:1500], y[1500:]

model = make_pipeline(
    StandardScaler(),
    LinearSVC('l1', dual=False)  # Task: use L1 regularization
)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

dummy = DummyClassifier()
dummy.fit(X_train, y_train)
dummy_score = dummy.score(X_test, y_test)

print(score)
print(dummy_score)
shape = X[0].shape
lsvc = model.steps[-1][-1]

# reshape weights to fit the shape of input
w = np.reshape(lsvc.coef_, shape)

# display what weights are applied to what features of input sequence
display(pd.DataFrame(w, columns=lumber.columns))
# first, load the data!
import pandas as pd

data = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv', sep='\t')
display(data.head())
# split inputs and outputs
X = data['review'].values
y = data['sentiment'].values

print(X[0])
print(y[0])
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py
from keras.preprocessing.text import Tokenizer

# proper sklearn TransformerMixin class
class TextToIntSeq(BaseEstimator, TransformerMixin):
    """ for a set of text, convert every text to sequence of num_words
    most frequent words, where a word is represented as integer. Words
    which are not frequent enough are replaced with 0.
    """
    def __init__(self, num_words=10000, max_seq_length=80):
        self.num_words = num_words
        self.max_seq_length = max_seq_length
        self._tokenizer = None
    
    def fit(self, X, y=None):
        # X: list of texts
        self._tokenizer = Tokenizer(self.num_words)
        self._tokenizer.fit_on_texts(X)
        return self  # proper sklearn transformer
    
    def transform(self, X, y=None):
        N = self.max_seq_length
        X = self._tokenizer.texts_to_sequences(X) # convert texts to sequences
        # trim sequences which are too long
        X = [x[:min(len(x), N)] for x in X]
        # add zeros for too small sequences
        X = [(N - len(x))*[0] + x for x in X]
        return np.array(X)
tok = TextToIntSeq()
tok.fit(X)
Xt = tok.transform(X)
print(Xt.shape)
print(tok.transform(np.array([
    'Hello world'
])))
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Dense, GRU, Embedding, Softmax
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam

X_train, X_test, y_train, y_test = train_test_split(Xt, y)

# definition of the network
inp = Input(shape=X_train[0].shape)

h = Embedding(tok.num_words, 128)(inp)
h = GRU(128)(h)  # Task: optimize number of neurons!
h = Dense(2)(h)  # only 2 classes are present
h = Softmax()(h)

model = Model(inputs=[inp], outputs=[h])

# try using different optimizers and different optimizer configs
model.compile(
    optimizer=Adam(), 
    loss=sparse_categorical_crossentropy,
    metrics=['accuracy']
)

model.fit(X_train, y_train, batch_size=64, epochs=3)
loss, acc = model.evaluate(X_test, y_test, batch_size=64)
print('Test score:', score)
print('Test accuracy:', acc)
my_input = tok.transform(np.array([
    'Best movie EVER!!!',
    'Worst movie EVER!!!'
]))

model.predict(my_input)