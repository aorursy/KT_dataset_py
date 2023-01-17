import pandas as pd

import spacy

import numpy as np



AUTHORS = { 'EAP' : 0, 'HPL' : 1, 'MWS' : 2 }



# Load SpaCy's models

SPACY = spacy.load('en')



# Load the training data

dataset = pd.read_csv("../input/train.csv")



# Convert the author strings into numbers

dataset['author'] = dataset['author'].apply(lambda x: AUTHORS[x])
sentence_lengths = np.fromiter((len(t.split()) for t in dataset['text']), count=len(dataset['text']), dtype='uint16')



print("Minimal sentence length {}: '{}'".format(

    np.min(sentence_lengths),

    dataset['text'][np.argmin(sentence_lengths)]

))



print("Maximal sentence length", np.max(sentence_lengths))
from sklearn.base import BaseEstimator, TransformerMixin



class PosPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, length_percentile = 95):

        self.length_percentile = length_percentile

        self._standartization_factor = 0



    def transform(self, X, *_):

        assert (self.sentence_size is not None), "Fitting required"

        

        # Create the output matrix

        result = np.zeros((len(X), self.sentence_size), dtype='uint8')

        

        # Tokenize and POS tag all the documents using multi-threading

        for i, x in enumerate(SPACY.pipe(X, batch_size=500, n_threads=-1)):

            # Store the POS-tags

            tags = np.fromiter((token.pos for token in x), dtype='uint8', count=len(x))

            

            # Pad and truncate data, if necessary, and store them in result

            last_index = len(tags) if len(tags) < self.sentence_size else self.sentence_size

            result[i, :last_index] = tags[:last_index]

        

        # Generate the factor one time to ensure applying the same factor at the next transformations

        if self._standartization_factor == 0:

            self._standartization_factor = np.min(result[result != 0]) - 1

    

        # Standartize all valid elements to count from 1

        result[result != 0] -= self._standartization_factor

        return result



    def fit(self, X, *_):

        # Define an optimal sentence size covering a specific percent of all sample

        self.sentence_size = int(np.percentile([len(t.split()) for t in X], self.length_percentile))

        return self

    

    def fit_transform(self, X, *_):

        self.fit(X)

        return self.transform(X)
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier



syntax_pipeline = Pipeline([

        ('pre', PosPreprocessor()), 

        ('predictor', RandomForestClassifier(n_estimators=100))

])



# Get a testing split for further tests

test_split = int(0.1 * len(dataset))



# Train the model and evaluate it on former unseen testing data

syntax_pipeline.fit(dataset['text'][:test_split], dataset['author'][:test_split])

syntax_pipeline.score(dataset['text'][test_split:], dataset['author'][test_split:])
from sklearn.base import BaseEstimator, ClassifierMixin

from keras.utils import to_categorical



from keras.models import Sequential

from keras.layers import Dense, LSTM, GRU, SimpleRNN, Activation, Dropout



class RnnClassifier(BaseEstimator, ClassifierMixin):



    def __init__(self,

                 batch_size=32,

                 epochs=3,

                 dropout=0,

                 rnn_type='gru',

                 hidden_layer=[64, 32]):

        

        # How many samples are processed in one training step?

        self.batch_size = batch_size

        # How long should the artificial neural network train?

        self.epochs = epochs

        # How much dropout do we put into the model to avoid overfitting?

        self.dropout = dropout

        # Which type of RNN do we want?

        self.rnn_type = rnn_type

        # Do we have hidden layer? If yes, how many which how many neurons?

        self.hidden_layer = hidden_layer

        

        self._rnn = None

        self._num_classes = None

        self._num_words = None



    def fit(self, X, Y=None):

        assert (Y is not None), "Y is required"

        assert (self.rnn_type in ['gru', 'lstm', 'simple']), "Invalid RNN type"



        # How many different tags do we have?

        self._num_words = np.max(X) + 1

        

        # How many classes should we predict?

        self._num_classes = np.max(Y) + 1

        

        node_type = None

        if self.rnn_type is 'gru':

            node_type = GRU

        elif self.rnn_type is 'lstm':

            node_type = LSTM

        else:

            node_type = SimpleRNN

        

        # Transfer the data into a appropiated shape

        X = self._reshape_input(X)



        # Ready for rumble?! Here the actual neural network starts!

        self._rnn = Sequential()

        self._rnn.add(node_type(X.shape[1], 

                                input_shape=(X.shape[1], self._num_words), 

                                return_sequences=(len(self.hidden_layer) > 0)

                               ))

        

        # Add dropout, if needed        

        if self.dropout > 0:

            self._rnn.add(Dropout(self.dropout))



        # Add the hidden layers and their dropout

        for (i, hidden_neurons) in enumerate(self.hidden_layer):

            sequences = i != len(self.hidden_layer) - 1

            

            self._rnn.add(node_type(hidden_neurons, return_sequences=sequences))

            if self.dropout > 0:

                self._rnn.add(Dropout(self.dropout))

        

        # Add the output layer and compile the model

        self._rnn.add(Dense(3, activation='softmax'))

        self._rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



        # Convert the results in the right format and start the training process

        Y = to_categorical(Y, num_classes=self._num_classes)

        self._rnn.fit(X, Y, epochs=self.epochs,

                      batch_size=self.batch_size,

                      verbose=0)



        return self



    def predict(self, X, y=None):

        if self._rnn is None:

            raise RuntimeError("Fitting required before prediction!")

        

        # 'Softmax' returns a list of probabilities - just use the highest onw

        return np.argmax(

            self._rnn.predict(

                self._reshape_input(X), 

                batch_size=self.batch_size

        ))



    def score(self, X, y=None):

        assert (y is not None), "Y is required"



        # Evaluate the model on training data

        return self._rnn.evaluate(

            self._reshape_input(X), 

            to_categorical(y, num_classes=self._num_classes)

        )[1]

    

    def _reshape_input(self, X):

        result = np.resize(X, (X.shape[0], X.shape[1], self._num_words))

        for x in range(0, X.shape[0]):

            for y in range(0, X.shape[1]):

                 result[x, y] = to_categorical(X[x, y], num_classes=self._num_words)[0]

        return result
from sklearn.model_selection import GridSearchCV, ShuffleSplit

from sklearn.pipeline import Pipeline



# Create the pipeline

syntax_pipeline = Pipeline([

        ('pre', PosPreprocessor()), 

        ('rnn', RnnClassifier(batch_size=64))

])



# Create the grid search with specifying possible parameter

optimizer = GridSearchCV(syntax_pipeline, {

    'rnn__rnn_type' : ('lstm', 'gru'), 

    'rnn__hidden_layer' : ([], [64], [64, 32]),

    'rnn__epochs': (30, 60, 90),

    'rnn__dropout': (0, 0.15, 0.3)

}, cv=ShuffleSplit(test_size=0.10, n_splits=1, random_state=0))



# Start the search: That will take some time!

# optimizer.fit(dataset['text'], dataset['author'])
from sklearn.model_selection import GridSearchCV, ShuffleSplit

from sklearn.pipeline import Pipeline



# Create the pipeline

syntax_pipeline = Pipeline([

        ('pre', PosPreprocessor()), 

        ('rnn', RnnClassifier(batch_size=64, rnn_type='gru', hidden_layer=[64]))

])



# Create the grid search with specifying possible parabeter

optimizer = GridSearchCV(syntax_pipeline, {

    'rnn__epochs': (25, 30, 35, 40, 45, 50),

    'rnn__dropout': (0.25, 0.3, 0.35, 0.4)

}, cv=ShuffleSplit(test_size=0.10, n_splits=1, random_state=0))



# Start the second search: Again, that will take some time!

# optimizer.fit(dataset['text'], dataset['author'])