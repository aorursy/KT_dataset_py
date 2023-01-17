import pandas as pd

import numpy as np 
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_id = train['id']

test_id = test['id']
train.drop(columns = ['id'], inplace = True)

test.drop(columns = ['id'], inplace = True)
train.isnull().sum()
test.isnull().sum()
train.drop(columns = ['keyword','location'], inplace = True)

test.drop(columns = ['keyword','location'], inplace = True)
train.head()
# Converting all text to lowercase

train['text'] = [t.lower() for t in train['text']]

test['text'] = [t.lower() for t in test['text']]
# Removing punctuations

import re

import string

train['text'] = [re.sub('[%s]' % re.escape(string.punctuation), '', i) for i in train['text']]

test['text'] = [re.sub('[%s]' % re.escape(string.punctuation), '', i) for i in test['text']]
# Removing numeric characters

train['text'] = [re.sub('\d','',n) for n in train['text']]

test['text'] = [re.sub('\d','',n) for n in test['text']]
import nltk

from nltk.tokenize import word_tokenize
# Word Tokenization



train['text'] = [word_tokenize(i) for i in train['text']]

test['text'] = [word_tokenize(i) for i in test['text']]
train['text'].head()
# Stop Words Removal



from nltk.corpus import stopwords



stop_words = set(stopwords.words('english'))

train['text'] = [[i for i in j if not i in stop_words] for j in train['text']]

test['text'] = [[i for i in j if not i in stop_words] for j in test['text']]
train.head()
from collections import defaultdict

from nltk.tag import pos_tag

from nltk.corpus import wordnet as wn



tag_map = defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ

tag_map['V'] = wn.VERB

tag_map['R'] = wn.ADV



tag_map
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()



train['text'] = [[lemmatizer.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(i)] for i in train['text']]

test['text'] = [[lemmatizer.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(i)] for i in test['text']]
train.head()
train['lemmatized_text'] = train['text'].apply(lambda x : ' '.join(x))

test['lemmatized_text'] = test['text'].apply(lambda x : ' '.join(x))
train.head()
train.drop(columns = ['text'], inplace = True)

test.drop(columns = ['text'], inplace = True)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features = 5000)



train_emb = tfidf.fit_transform(train['lemmatized_text']).toarray()

test_emb = tfidf.fit_transform(test['lemmatized_text']).toarray()
train_emb.shape[1:]
y = train['target']
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
MNB = MultinomialNB()
x_train,x_valid,y_train,y_valid = train_test_split(train_emb,y,test_size = 0.3, random_state = 100) 
MNB.fit(x_train,y_train)

pred_MNB = MNB.predict(x_valid)
print("Accuracy score : {:.2f}".format(accuracy_score(y_valid, pred_MNB)))
print("ROC-AUC score : {:.2f}".format(roc_auc_score(y_valid, pred_MNB)))
print(classification_report(y_valid, pred_MNB))
MNB.fit(train_emb,y)
MNB_predictions = MNB.predict(test_emb)
Prediction_results = pd.DataFrame({"target": MNB_predictions}, index = test_id)
#submission_file = Prediction_results.to_csv('submission.csv')
from sklearn import svm

SVC = svm.SVC()

#SVC.fit(x_train,y_train)

#pred_SVC = SVC.predict(x_valid)
#print("Accuracy score : {:.2f}".format(accuracy_score(y_valid, pred_SVC)))
#print("ROC-AUC score : {:.2f}".format(roc_auc_score(y_valid, pred_SVC)))
from collections import Counter



# Finding the number of unique word in the corpus

def word_counter(text):

    count = Counter()

    for i in text.values:

        for word in i.split():

            count[word] += 1

    return count
train.head()
train_text = train.lemmatized_text

counter = word_counter(train_text)

counter
print("Number of unique words in the corpus : {:.2f}".format(len(counter)))
words = len(counter)

# maximum number of words in a sequence

max_length = 20
train_sent = train['lemmatized_text']

train_labels = train['target']

test_sent = test['lemmatized_text']
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=words)

tokenizer.fit_on_texts(train_sent)
word_index = tokenizer.word_index

word_index
train_sequence = tokenizer.texts_to_sequences(train_sent)
train_sequence[0]
test_sequence = tokenizer.texts_to_sequences(test_sent)
test_sequence
from keras.preprocessing.sequence import pad_sequences



train_padded = pad_sequences(train_sequence, maxlen = max_length, padding = "post", truncating = "post")
train_padded
test_padded = pad_sequences(test_sequence, maxlen = max_length, padding = "post", truncating = "post")
test_padded
from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.initializers import Constant

from keras.optimizers import Adam

import tensorflow as tf



def leaky_relu(z, name = None):

    return tf.maximum(0.01*z,z, name = name)



model = Sequential()



model.add(Embedding(words,32,input_length = max_length))

#model.add(LSTM(128, return_sequences = True, dropout = 0.1))

model.add(LSTM(64, dropout = 0.1))

model.add(Dense(units = 32 , activation = leaky_relu))

model.add(Dense(1, activation = tf.nn.elu))



optimizer = Adam(learning_rate = 3e-4)



model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.summary()
model.fit(train_padded, train_labels, epochs = 40)
import h5py

#model.save('baseline_lstm_model.h5')
from keras.models import load_model

#model = load_model('baseline_lstm_model.h5')
lstm_base_pred = model.predict_classes(test_padded, verbose = 0)
lstm_base_pred = lstm_base_pred.reshape(-1,1).ravel()
len(lstm_base_pred)
Prediction_results_lstm = pd.DataFrame({"target":lstm_base_pred}, index = test_id)

Prediction_results_lstm
#submission_lstm_elu_leaky_relu = Prediction_results_lstm.to_csv('submission_lstm_elu_leaky_relu.csv')