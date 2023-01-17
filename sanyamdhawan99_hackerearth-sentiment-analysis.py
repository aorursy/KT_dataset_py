import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout, Flatten, TimeDistributed, BatchNormalization, SpatialDropout1D
from keras.layers.wrappers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from gensim.models.keyedvectors import KeyedVectors

from sklearn.metrics import classification_report, confusion_matrix, f1_score

import tensorflow as tf
import math
import warnings
from tqdm import tqdm
import scipy
import statistics 
# train_df_master = pd.read_csv('../input/complete-data-for-sentiment-analysis/Complete Data for Sentiment Analysis.csv')
# train_df_master = pd.read_csv('../input/originaldataset/Train_data_cleaned.csv')
train_df_master = pd.read_csv('../input/originaldatasetcombined/Combined_Train_data_cleaned.csv')
test_df_master = pd.read_csv('../input/test-data/Test_data_cleaned.csv')
train_df = train_df_master.copy()
print('Training data has {} rows and {} columns'.format(train_df.shape[0], train_df.shape[1]))

test_df = test_df_master.copy()
print('Testing data has {} rows and {} columns'.format(test_df.shape[0], test_df.shape[1]))
sns.countplot(train_df['sentiment_class']).set_title('Sentiment Class distribution')
plt.show()
# define the training and testing lengths to keep a track on encoded data
train_length = len(train_df)
test_length = len(test_df)
""" Although the data in this file is already cleaned, we still perform a double check """

def tokenize(text):
    stop_words = stopwords.words('english')    
    
    tokenized_text = []
    
    for txt in text:
        txt = str(txt)
        words = txt.split(' ')
        tokenized_string = ''

        for word in words:
            # check for name handles and unwanted text
            if len(word) > 1 and word[0] != '@' and word not in stop_words:
                # if the word is a hastag, remove #
                if word[0] == '#':
                    word = word[1:]

                tokenized_string += word + ' '

        tokenized_text.append(tokenized_string)
    
    return tokenized_text
""" encode text -> translate text to a sequence of numbers """

def encode_text(text):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ", lower=True)
    tokenizer.fit_on_texts(text)
    
    return tokenizer, tokenizer.texts_to_sequences(text)
""" Apply padding to dataset and convert labels (-1, 0, 1) to bitmaps """

def format_data(encoded_text, max_length, sent_labels):
    x = pad_sequences(encoded_text, maxlen= max_length, padding='post')
    y = []
    
    for label in sent_labels:
        bit_vec = np.zeros(3)
        bit_vec[label+1] = 1
        y.append(bit_vec)
        
    y = np.asarray(y)
    return x, y
""" create weight matrix from pre trained embeddings """

def create_weight_matrix(vocab, raw_embeddings, dim):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, dim))
    
    for word, idx in vocab.items():
        if word in raw_embeddings:
            weight_matrix[idx] = raw_embeddings[word]
    
    return weight_matrix
## 43.84

""" final model 1"""
def final_model_1(weight_matrix, vocab_size, max_length):
    embedding_layer = Embedding(vocab_size, 50, weights=[weight_matrix], input_length=max_length, trainable=True, 
                                mask_zero=True)
    model = Sequential()
    model.add(embedding_layer)
    
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(64, activation='softmax'))
    model.add(Dense(64, activation='softmax'))
    
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
## 42.69292

""" final model 2"""
def final_model_2(weight_matrix, vocab_size, max_length):
    embedding_layer = Embedding(vocab_size, 300, weights=[weight_matrix], input_length=max_length, trainable=True, 
                                mask_zero=True)
    model = Sequential()
    model.add(embedding_layer)
    
    model.add(Bidirectional(LSTM(256, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    
    model.add(Dense(64, activation = 'softmax'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
# 42.57 score

""" final model 3"""
def final_model_3(weight_matrix, vocab_size, max_length):
    embedding_layer = Embedding(vocab_size, 50, weights=[weight_matrix], input_length=max_length, trainable=True, 
                                mask_zero=True)
    model = Sequential()
    model.add(embedding_layer)
    
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
""" Tokenize all the training and testing text """

tokenized_text = tokenize(train_df['text'])
tokenized_text += tokenize(test_df['text'])

max_length = math.ceil(sum([len(s.split(" ")) for s in tokenized_text])/len(tokenized_text))

tokenizer, encoded_text = encode_text(tokenized_text)

max_length, len(tokenized_text)
""" Apply padding and format data """

x, y = format_data(encoded_text[:train_length], max_length, train_df['sentiment_class'])
print('For train data: ', len(x), len(y))

x_test = pad_sequences(encoded_text[train_length:], maxlen= max_length, padding='post')
print('For test data: ', len(x_test))
""" Clearing vocabulary """

vocab = tokenizer.word_index
vocab, len(vocab)
# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# load the GloVe vectors in a dictionary:

embeddings_index_50 = {}
f = open('/kaggle/input/glove6b50dtxt/glove.6B.50d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index_50[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index_50))
""" Create the weight matrix """

weight_matrix_300 = create_weight_matrix(vocab, embeddings_index, dim = 300)
len(weight_matrix_300)
""" Create the weight matrix """

weight_matrix_50 = create_weight_matrix(vocab, embeddings_index_50, dim = 50)
len(weight_matrix_50)
model1 = final_model_1(weight_matrix_50, len(vocab)+1, max_length)

history1 = model1.fit(x, y, epochs = 20, validation_split = 0.1, verbose = 1)
model1.summary()
model2 = final_model_2(weight_matrix_300, len(vocab)+1, max_length)

history2 = model2.fit(x, y, epochs = 20, validation_split = 0.1, verbose = 1)
model2.summary()
model3 = final_model_3(weight_matrix_50, len(vocab)+1, max_length)

history3 = model3.fit(x, y, epochs = 30, validation_split = 0.1, verbose = 1)
model3.summary()
train_pred_prob_1 = model1.predict(x)
train_pred_prob_1
train_pred_1 = [np.argmax(pred)-1 for pred in train_pred_prob_1]
len(train_pred_1)
train_pred_prob_2 = model2.predict(x)
train_pred_prob_2
train_pred_2 = [np.argmax(pred)-1 for pred in train_pred_prob_2]
len(train_pred_2)
train_pred_prob_3 = model3.predict(x)
train_pred_prob_3
train_pred_3 = [np.argmax(pred)-1 for pred in train_pred_prob_3]
len(train_pred_3)
train_pred_comb = np.concatenate((train_pred_1, train_pred_2, train_pred_3), axis = 0).reshape(-1, len(train_pred_1))
print(train_pred_comb)
train_pred_comb[:, 0]
final_pred = []
for i in range(len(train_pred_1)):
    try:
        final_pred.append(statistics.mode(train_pred_comb[:, i]))
    except:
        final_pred.append(train_pred_1[i])

final_pred[:5]
train_pred = final_pred
train_f1_score = f1_score(train_df['sentiment_class'], train_pred, average='weighted')

print('F1 Score: ', train_f1_score)
print()
print('Confusion Matrix: \n', confusion_matrix(train_df['sentiment_class'], train_pred))
print()
print('Classification Report: \n', classification_report(train_df['sentiment_class'], train_pred))
test_pred_prob_1 = model1.predict(x_test)
test_pred_prob_2 = model2.predict(x_test)
test_pred_prob_3 = model3.predict(x_test)
test_pred_1 = [np.argmax(pred)-1 for pred in test_pred_prob_1]
test_pred_2 = [np.argmax(pred)-1 for pred in test_pred_prob_2]
test_pred_3 = [np.argmax(pred)-1 for pred in test_pred_prob_3]

test_pred_combined = np.concatenate((test_pred_1, test_pred_2, test_pred_3), axis = 0).reshape(-1, len(test_pred_1))
final_pred = []
for i in range(len(test_pred_1)):
    try:
        final_pred.append(statistics.mode(test_pred_combined[:, i]))
    except:
        final_pred.append(test_pred_1[i])

final_pred[:5]
test_pred = final_pred
org_test_df = pd.read_csv('../input/original-dataset/test.csv')
org_test_df.head()
submission_df = org_test_df.copy()
submission_df.drop(['original_text', 'lang', 'retweet_count', 'original_author'], axis = 1, inplace = True)
submission_df['sentiment_class'] = test_pred

print(submission_df['sentiment_class'].value_counts())
sns.countplot(submission_df['sentiment_class']).set_title('Class distribution')
submission_df.to_csv('checking model for submissions final.csv', index = False)
submission_df.head()
submission_df.head(20)
