# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Embedding, Dropout, GlobalAveragePooling1D, LSTM, \
Conv1D, MaxPool1D, SpatialDropout1D, Bidirectional, GRU, Input, BatchNormalization
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from string import punctuation
from xgboost import XGBClassifier
from tqdm.notebook import tqdm
import plotly.express as px
import joblib


from transformers import BertTokenizer, TFBertModel, DistilBertTokenizerFast, TFDistilBertModel

#import kaggle

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
punct = dict.fromkeys(punctuation, ' ')
tok = TweetTokenizer()
#glove_embd = joblib.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl')
glove_embd = np.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', allow_pickle= True)
data = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv', encoding='utf8')
data.drop(columns=['id'], inplace=True)
data.fillna(' ', inplace=True)
data['text'] = data['keyword'].astype(str) + ' ' + data['location'].astype(str) + ' ' + data['text'].astype(str)

#data2 = data.copy()
#data = pd.concat([data,data2])
def comments_processor(data):
    corpus = []
    data = data.lower()
    data = re.sub(pattern=r'(((http)(s)?|www(.)?)(://)?\S+)', repl='', string=data)
    data = re.sub(pattern=r'\d', repl='', string=data)
    data = str.translate(data, str.maketrans(punct))
    data = re.sub(pattern=r'\s{2,}', repl=' ', string=data)
    data = tok.tokenize(data)
    data = [item for item in data if item not in stopwords]
    data = ' '.join([item for item in data if item not in stopwords])
    
    return data
data['text'] = data['text'].apply(comments_processor)
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'].values, train_size = 0.85,\
                                                    random_state = 11, stratify = data['target'])
count_vec = CountVectorizer(max_df = 0.8, min_df = 5)
count_vec.fit_transform(X_train)
%%time
clf = Pipeline([('vect', count_vec),\
                ('classifier', RandomForestClassifier( n_jobs=-1, random_state=33))])
clf.fit(X_train, np.array(y_train))
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
%%time
clf2 = Pipeline([('vect', count_vec),\
                ('classifier', XGBClassifier(n_jobs=-1, random_state=33, eta = 0.9, max_delta_step = 5,gamma = 1))])

clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
accuracy_score(y_test, y_pred) 
%%time
clf3 = Pipeline([('vect', count_vec),\
                ('classifier', SVC(random_state=33))])

clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
accuracy_score(y_test, y_pred) 
pd.DataFrame.from_dict(classification_report(y_test, y_pred, output_dict=True)).T
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
punct = dict.fromkeys(punctuation, ' ')
tok = TweetTokenizer()
data = pd.read_csv(r'/kaggle/input/nlp-getting-started/train.csv', encoding='utf8')
data.drop(columns=['id'], inplace=True)
data.fillna(' ', inplace=True)
data['text'] = data['keyword'].astype(str) + ' ' + data['location'].astype(str) + ' ' + data['text'].astype(str)
max_len = 50
embed_dim = 300
pad_type = 'post'
trun_type = 'post'
def comments_processor(data):
    corpus = []
    data = data.lower()
    data = re.sub(pattern=r'(((http)(s)?|www(.)?)(://)?\S+)', repl='', string=data)
    data = re.sub(pattern=r'\d', repl='', string=data)
    data = re.sub(pattern=r"n't", repl=' not', string=data)
    data = str.translate(data, str.maketrans(punct))
    data = re.sub(pattern=r'\s{2,}', repl=' ', string=data)
    data = tok.tokenize(data)
    data = [item for item in data if item not in stopwords]
    corpus.extend(data)
    
    return corpus
    
corpus = [' '.join(corpus) for corpus in data['text'].apply(comments_processor)]
tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index 

corpus_seq = tokenizer.texts_to_sequences(corpus)
corpus_padded = pad_sequences(corpus_seq, maxlen= max_len, padding= pad_type, truncating= trun_type)
len(word_index)
num_words = len(word_index)+1

embedding_matrix = np.zeros((num_words,300))
oov = dict()
for word, i in tqdm(word_index.items()):
    if i > num_words:
        continue
    try:
        embed_vec = glove_embd[word]
    except:
        continue
    if embed_vec is not None:
       embedding_matrix[i] =  embed_vec
X_train, X_test, y_train, y_test = train_test_split(corpus_padded, data['target'].values, train_size = 0.85,\
                                                    random_state = 11, stratify = data['target'])
train_padded = np.array(X_train)
y_train = np.array(y_train)
test_padded = np.array(X_test)
y_test = np.array(y_test)
model = Sequential()
embedding = Embedding(num_words, embed_dim, embeddings_initializer = Constant(embedding_matrix),\
                      input_length = max_len, trainable = False)
model.add(embedding)
model.add(SpatialDropout1D(0.3))
model.add(Conv1D(128, 8))
model.add(MaxPool1D())
model.add(SpatialDropout1D(0.3))
model.add(Flatten())
model.add(Dense(24, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer =\
              tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), metrics = ['accuracy'])
model.summary()
history = model.fit(train_padded, y_train, batch_size = 4, epochs = 12,\
                    validation_data = (test_padded, y_test), verbose = 2)
model_rnn = Sequential()
embedding = Embedding(num_words, embed_dim, embeddings_initializer = Constant(embedding_matrix),\
                      input_length = max_len, trainable = False)
model_rnn.add(embedding)
model_rnn.add(SpatialDropout1D(0.3))
model_rnn.add(Bidirectional(LSTM(64, recurrent_dropout = 0.2, return_sequences = True)))
model_rnn.add(Dropout(0.2))
model_rnn.add(LSTM(32, recurrent_dropout = 0.2))
model_rnn.add(Flatten())
model_rnn.add(Dropout(0.2))
model_rnn.add(Dense(32, activation = 'relu')) #256
model_rnn.add(Dense(1, activation = 'sigmoid'))

model_rnn.compile(loss = 'binary_crossentropy', optimizer = \
                  tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), metrics = ['accuracy'])
model_rnn.summary()
history_rnn = model_rnn.fit(train_padded, y_train, batch_size = 4, epochs = 10,\
                            validation_data = (test_padded, y_test), verbose = 2)
model_gru = Sequential()
embedding = Embedding(num_words, embed_dim, embeddings_initializer = Constant(embedding_matrix),\
                      input_length = max_len, trainable = False)
model_gru.add(embedding)
model_gru.add(SpatialDropout1D(0.2))
model_gru.add(Bidirectional(GRU(64, recurrent_dropout = 0.2, return_sequences = True)))
model_gru.add(Dropout(0.2))
model_gru.add(Bidirectional(GRU(32, recurrent_dropout = 0.2)))
model_gru.add(Flatten())
model_gru.add(Dropout(0.2))
model_gru.add(Dense(32, activation = 'relu'))
model_gru.add(Dense(1, activation = 'sigmoid'))

model_gru.compile(loss = 'binary_crossentropy', optimizer = \
                  tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), metrics = ['accuracy'])
model_gru.summary()
history_gru = model_gru.fit(train_padded, y_train, batch_size = 4, epochs = 10, validation_data = (test_padded, y_test), verbose = 2)
test_data = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv', encoding='utf8')
test_data.fillna(' ', inplace=True)
test_data['text'] = test_data['keyword'].astype(str) + ' ' + test_data['location'].astype(str) + ' ' + test_data['text'].astype(str)
test_corpus = [' '.join(corpus) for corpus in test_data['text'].apply(comments_processor)]
test_corpus_seq = tokenizer.texts_to_sequences(test_corpus)
test_corpus_pad = pad_sequences(test_corpus_seq, maxlen= max_len, truncating=trun_type, padding= pad_type)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
distil_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

distil_bert_layer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
max_len = 100
def tweek_tokenizer(tweets, tokenizer ,max_len = max_len):
    input_ids = []
    
    
    for text in tweets:
        
        text = re.sub(pattern=r'(((http)(s)?|www(.)?)(://)?\S+)', repl='', string=text)
        #text = re.sub(pattern=r'([^a-zA-Z0-9\s])', repl=' ', string=text)
        text = re.sub(pattern=r'\s{2,}', repl=' ', string=text)
        
        tokenized_text = tokenizer.encode_plus(text, add_special_tokens = True, max_length = max_len, \
                                               pad_to_max_length = True, return_tensors = 'tf')   
        
        input_ids.append(tokenized_text['input_ids'])    
    
    input_ids = np.array(input_ids).reshape([-1,max_len])
    
    return input_ids
def bert_model(bert_layer, max_len = max_len):
    
    input_tokens = Input(shape=(max_len), dtype= tf.int32, name='input_ids')
    
    bert_out = bert_layer([input_tokens])[0] #, token_type_ids, attention_mask 
    
    bert_out = tf.keras.layers.Lambda(lambda seq : seq [:,0,:])(bert_out)
    
    bert_out = Dropout(0.5)(bert_out)
    
    dense_1 = Dense(768, activation = 'tanh', kernel_regularizer = tf.keras.regularizers.L2(l2= 0.1))(bert_out)
    
    drop_output = Dropout(0.3)(dense_1)

    dense_out = Dense(1, activation = 'sigmoid', name= 'output2')(drop_output)
     
    model = Model(inputs = input_tokens, outputs = dense_out)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=2e-5, decay = 0.01), \
                  loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model
%%time
input_ids = tweek_tokenizer(data['text'].values, tokenizer ,max_len=max_len)
model = bert_model(bert_layer)
model.summary()
model.fit(x= [input_ids], y= np.array(data['target'].values), batch_size = 16,epochs= 3, shuffle= True,validation_split= 0.2)
test_data = pd.read_csv(r'/kaggle/input/nlp-getting-started/test.csv', encoding='utf8')
test_data.fillna(' ', inplace=True)
test_data['text'] = test_data['keyword'].astype(str) + ' ' + test_data['location'].astype(str) + ' ' + test_data['text'].astype(str)
%%time
test_input_ids= tweek_tokenizer(test_data['text'].values, tokenizer ,max_len=max_len)
%%time
test_pred = model.predict(test_input_ids)
test_pred = np.round(test_pred).astype(int)
test_pred = test_pred.reshape(3263)
sample_submission = pd.DataFrame({'id' : test_data['id'].values.tolist(), 'target' : test_pred})
sample_submission.to_csv(r'sample_submission.csv', encoding='utf8', index= False, header = True)