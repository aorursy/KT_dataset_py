# Data Processing
import numpy as np 
from numpy import asarray
import pandas as pd 
from tqdm import tqdm #TQDM is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.
from sklearn.model_selection import train_test_split
import re

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tokenization

# Data Modeling
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer # == CountVectorizer + TfidfTransformer
from sklearn.svm import LinearSVC

from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

import tensorflow as tf
import tensorflow_hub as hub


# Data Evaluation
from sklearn import metrics


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
!unzip ../input/sentiment-analysis-on-movie-reviews/train.tsv.zip 
!unzip ../input/sentiment-analysis-on-movie-reviews/test.tsv.zip
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')
def remove_URL(text):
    #url = re.compile(r'https?://\S+|www\.\S+')
    url = re.compile(r'http\S+|www.\S+')  # https / http / www
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>') # '<p>string<p>' -> 'string'
    #html=re.compile(r'<.*>') # '<p>string<p>' -> ''
    return html.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_number(text):
    url = re.compile(r'[0-9]')  
    return url.sub(r'',text)

def remove_non_alphabet(text):
    url = re.compile(r'[^a-z\s]')  
    return url.sub(r' ',text) # space is handled by Tokenizer of Keras, don't worry

import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
train['Phrase'] = train['Phrase'].str.lower()
train['Phrase'] = train['Phrase'].transform(remove_URL)
train['Phrase'] = train['Phrase'].transform(remove_html)
train['Phrase'] = train['Phrase'].transform(remove_emoji)
train['Phrase'] = train['Phrase'].transform(remove_number)
train['Phrase'] = train['Phrase'].transform(remove_non_alphabet)

test['Phrase'] = test['Phrase'].str.lower()
test['Phrase'] = test['Phrase'].transform(remove_URL)
test['Phrase'] = test['Phrase'].transform(remove_html)
test['Phrase'] = test['Phrase'].transform(remove_emoji)
test['Phrase'] = test['Phrase'].transform(remove_number)
test['Phrase'] = test['Phrase'].transform(remove_non_alphabet)
X = train['Phrase']
y = train['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.25, random_state=0)
# All steps at once
text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)

# Form a prediction set
y_pred = text_clf.predict(X_test)

# Print the overall accuracy
print('LinearSVC Score: ', metrics.accuracy_score(y_test,y_pred))
tokenize = Tokenizer()
tokenize.fit_on_texts(X_train.values)

vocab_size = len(tokenize.word_index) + 1

X_train = tokenize.texts_to_sequences(X_train)
X_test = tokenize.texts_to_sequences(X_test)
#X_test = tokenize.texts_to_sequences(test['Phrase'])

max_lenght = max([len(s.split()) for s in train['Phrase']])

X_train = pad_sequences(X_train, max_lenght, padding='post')
X_test = pad_sequences(X_test, max_lenght, padding='post')
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=max_lenght))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2 ))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train,validation_data=(X_test, y_test), batch_size=128, epochs=4, verbose=1)
results_Embedding = model.evaluate(X_test, y_test, batch_size=128)
print('Embedding Test Accuracy Score: ', results_Embedding[1])
# load the whole embedding into memory
embeddings_index = dict()
f = open('../input/glove6b100dtxt1/glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenize.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_lenght))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2 ))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train,validation_data=(X_test, y_test), batch_size=128, epochs=7, verbose=1)
# results_GloVe = model.evaluate(X_test, y_test, batch_size=128)
# print('GloVe Test Accuracy Score: ', results_GloVe[1])
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(5, activation='softmax')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
max_len = 52
train_input = bert_encode(train['Phrase'].values, tokenizer, max_len=max_len)
#train_labels = tf.keras.utils.to_categorical(train['Sentiment'].values, num_classes=5)
train_labels = train['Sentiment'].values
model = build_model(bert_layer, max_len=max_len)
model.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', 
                                                save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
train_history = model.fit(
    train_input, train_labels, 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint, earlystopping],
    batch_size=32)
# text_clf.fit(X, y)

# # Form a prediction set
# y_pred = text_clf.predict(test['Phrase'])

# sub = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')
# sub['Sentiment'] = y_pred
# sub.to_csv('submission.csv', index=False)


# y_pred = model.predict(X_test)

# sub = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')
# sub['Sentiment'] = np.argmax(y_pred, axis=-1)
# sub.to_csv('submission.csv', index=False)


# y_pred = model.predict_classes(X_test)
# sub = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')
# sub['Sentiment'] = y_pred
# sub.to_csv('submission.csv', index=False)