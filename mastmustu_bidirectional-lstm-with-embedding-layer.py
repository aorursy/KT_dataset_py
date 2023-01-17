# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('Loading in comments...')



train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train['comment_length'] = train['text'].apply(lambda x : len(x))

train['comment_length'].hist()
from sklearn.base import BaseEstimator, TransformerMixin

from keras.models import Model, Input

from keras.layers import Dense, LSTM, Dropout, Embedding, SpatialDropout1D, Bidirectional, concatenate

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score

from eli5.lime import TextExplainer

import regex as re

import numpy as np

import eli5





class KerasTextClassifier(BaseEstimator, TransformerMixin):

    '''Wrapper class for keras text classification models that takes raw text as input.'''

    

    def __init__(self, max_words=30000, input_length=100, emb_dim=20, n_classes=2, 

                 epochs=5, batch_size=32):

        self.max_words = max_words

        self.input_length = input_length

        self.emb_dim = emb_dim

        self.n_classes = n_classes

        self.epochs = epochs

        self.bs = batch_size

        self.model = self._get_model()

        self.tokenizer = Tokenizer(num_words=self.max_words+1,

                                   lower=True, split=' ', oov_token="UNK")

    

    def _get_model(self):

        input_text = Input((self.input_length,))

        text_embedding = Embedding(input_dim=self.max_words + 2, output_dim=self.emb_dim,

                                   input_length=self.input_length, mask_zero=False)(input_text)

        text_embedding = SpatialDropout1D(0.5)(text_embedding)

        bilstm = Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.5))(text_embedding)

        x = concatenate([GlobalAveragePooling1D()(bilstm), GlobalMaxPooling1D()(bilstm)])

        x = Dropout(0.7)(x)

        x = Dense(512, activation="relu")(x)

        x = Dropout(0.6)(x)

        x = Dense(512, activation="relu")(x)

        x = Dropout(0.5)(x)

        out = Dense(units=self.n_classes, activation="softmax")(x)

        model = Model(input_text, out)

        model.compile(optimizer="adam",

                      loss="sparse_categorical_crossentropy",

                      metrics=["accuracy"])

        return model

    

    def _get_sequences(self, texts):

        seqs = self.tokenizer.texts_to_sequences(texts)

        return pad_sequences(seqs, maxlen=self.input_length, value=0)

    

    def _preprocess(self, texts):

        return [re.sub(r"\d", "DIGIT", x) for x in texts]

    

    def fit(self, X, y):

        '''

        Fit the vocabulary and the model.

        

        :params:

        X: list of texts.

        y: labels.

        '''

        

        self.tokenizer.fit_on_texts(self._preprocess(X))

        self.tokenizer.word_index = {e: i for e,i in self.tokenizer.word_index.items() if i <= self.max_words}

        self.tokenizer.word_index[self.tokenizer.oov_token] = self.max_words + 1

        seqs = self._get_sequences(self._preprocess(X))

        self.model.fit(seqs, y, batch_size=self.bs, epochs=self.epochs, validation_split=0.1)

    

    def predict_proba(self, X, y=None):

        seqs = self._get_sequences(self._preprocess(X))

        return self.model.predict(seqs)

    

    def predict(self, X, y=None):

        return np.argmax(self.predict_proba(X), axis=1)

    

    def score(self, X, y):

        y_pred = self.predict(X)

        return accuracy_score(y, y_pred)
text_model = KerasTextClassifier(epochs=5, max_words=20000, input_length=160)
train.columns
text_model.fit(train.text, train.target)
text_model.score(train.text , train.target)
print('Loading in comments...')



test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

pred = text_model.predict(test.text)

test['pred'] = pred 





print('Loading in Submission File...')



submit_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submit_df['target'] = test['pred']



submit_df.to_csv('5E_submit.csv', index=False)