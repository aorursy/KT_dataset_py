import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback, EarlyStopping
from keras import regularizers
from sklearn.metrics import f1_score
from gensim.models.keyedvectors import KeyedVectors
hyper_params = {
    'validation_split': 0.01,
    'batch_size': 64,
    'sample_ratio': 1.0,
    'num_words': 40000,
    'epochs': 10,
    'embedding_size': 300,
    'keep_probability': 0.75,
    'l2_regularization': 0.01,
    'lstm_size': 50,
    'dense_size': 50,
    'max_sequence': 60,
    'sampling_size': -1,
    'min_length': 1,
    'max_length': 50
}

config = {
    'lowercase': True,
    'stemming': True,
    'remove_stopwords': True,
    'remove_non_letters': True,
    'remove_punctuation': True,
    'reduce_lengthening': True,
    'sort': True,
    'trim': True,
    'early_stopping': False,
    'sub_sampling': True
}
train = pd.read_csv('../input/train.csv')
if hyper_params['sampling_size'] > 0:
    train = train.sample(hyper_params['sampling_size'])
if config['sub_sampling']:
    negative_df = train[train['target'] == 0]
    positive_df = train[train['target'] == 1]
    positive = int((len(train) - len(negative_df)) * hyper_params['sample_ratio'])
    train = pd.concat([negative_df.sample(positive, random_state=42), positive_df])
test = pd.read_csv('../input/test.csv')
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def reduce_lengthening(data):
    length_regex = re.compile(r"(.)\1{2,}")
    return [length_regex.sub(r"\1\1", x) for x in data['question_text']]

def remove_non_letters(data):
    letters_regex = re.compile('[^a-zA-Z ]')
    return [letters_regex.sub(' ', x) for x in data['question_text']]

def remove_stopwords(data):
    stopwords_set = set(stopwords.words('english'))
    temp = [[y if y not in stopwords_set else '' for y in x] for x in data['question_text']]
    return [filter(None, x) for x in temp]

def stemming(data):
    ps = PorterStemmer()
    temp = [[str(ps.stem(y)) for y in x.split(' ')] for x in data['question_text']]
    return [' '.join(x) for x in temp]

def remove_punctuation(data):
    return [x.translate(string.punctuation) for x in data['question_text']]

def lowercase(data):
    return [x.lower() for x in data['question_text']]

if config['lowercase']:
    train['question_text'] = lowercase(train)
    test['question_text'] = lowercase(test)
if config['remove_non_letters']:
    train['question_text'] = remove_non_letters(train)
    test['question_text'] = remove_non_letters(test)
if config['reduce_lengthening']:
    train['question_text'] = reduce_lengthening(train)
    test['question_text'] = reduce_lengthening(test)
if config['remove_punctuation']:
    train['question_text'] = remove_punctuation(train)
    test['question_text'] = remove_punctuation(test)
if config['stemming']:
    train['question_text'] = stemming(train)
    test['question_text'] = stemming(test)
if config['trim']:
    train = train[train.apply(lambda x: len(x['question_text'].split(' ')) >= hyper_params['min_length'] and 
                              len(x['question_text'].split(' ')) <= hyper_params['max_length'], axis=1)]
if config['sort']:
    train = train.reindex(train.question_text.str.len().sort_values().index)
    train = train.reset_index(drop=True)
train_sentences = train['question_text'].values
test_sentences = test['question_text'].values
tokenizer = Tokenizer(num_words=hyper_params['num_words'])
tokenizer.fit_on_texts(train_sentences)
train_tokenized = tokenizer.texts_to_sequences(train_sentences)
test_tokenized = tokenizer.texts_to_sequences(test_sentences)
word_index = tokenizer.word_index
X_train = sequence.pad_sequences(train_tokenized, maxlen=hyper_params['max_sequence'])
y_train = train.target.values
X_test = sequence.pad_sequences(test_tokenized, maxlen=hyper_params['max_sequence'])
word_vectors = KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
vocabulary_size = min(len(word_index) + 1, hyper_params['num_words'])
embedding_matrix = np.zeros((vocabulary_size, hyper_params['embedding_size']))
for word, i in word_index.items():
    if i>= hyper_params['num_words']:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), hyper_params['embedding_size'])
del(word_vectors)
class Metrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        print(" — f1_score: %f" % f1_score(val_targ, val_predict))
metrics = Metrics()
early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
model = Sequential()
model.add(Embedding(vocabulary_size, hyper_params['embedding_size'], weights=[embedding_matrix], trainable=True))
model.add(Bidirectional(LSTM(hyper_params['lstm_size'], dropout=1 - hyper_params['keep_probability'])))
model.add(Dense(hyper_params['dense_size'], activation='relu', 
                kernel_regularizer=regularizers.l2(hyper_params['l2_regularization'])))
model.add(Dropout(1 - hyper_params['keep_probability']))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
callbacks = [metrics]
if config['early_stopping']:
    callbacks += early_stopping
history = model.fit(X_train, y_train, batch_size=hyper_params['batch_size'], epochs=hyper_params['epochs'], validation_split=hyper_params['validation_split'], 
                    callbacks=callbacks)
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.prediction = model.predict_classes(X_test)
sample_submission.to_csv('submission.csv', index=False)