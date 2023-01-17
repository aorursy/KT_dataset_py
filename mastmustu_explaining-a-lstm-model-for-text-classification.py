from sklearn.datasets import fetch_20newsgroups



categories = ['alt.atheism', 'soc.religion.christian',

              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(

    subset='train',

    categories=categories,

    shuffle=True,

    random_state=42,

    remove=('headers', 'footers'),

)

twenty_test = fetch_20newsgroups(

    subset='test',

    categories=categories,

    shuffle=True,

    random_state=42,

    remove=('headers', 'footers'),

)
print(twenty_train.DESCR)

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

    

    def __init__(self, max_words=30000, input_length=100, emb_dim=20, n_classes=4, epochs=5, batch_size=32):

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

        bilstm = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.5))(text_embedding)

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
text_model = KerasTextClassifier(epochs=15, max_words=25000, input_length=200)

text_model.fit(twenty_train.data, twenty_train.target)

text_model.score(twenty_test.data, twenty_test.target)



doc = twenty_test.data[40]

te = TextExplainer(random_state=42)

te.fit(doc, text_model.predict_proba)

te.show_prediction(target_names=twenty_train.target_names)