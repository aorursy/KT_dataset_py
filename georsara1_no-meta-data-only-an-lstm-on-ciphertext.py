#Import needed libraries
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import itertools
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Flatten, Dropout, Convolution1D, Bidirectional, LSTM, CuDNNLSTM
from keras.layers.embeddings import Embedding
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_submission = pd.read_csv('../input/sample_submission.csv')
print("Train shape : ",df_train.shape)
print("Test shape : ",df_test.shape)
#Tokenize sentences
tokenizer = Tokenizer()

text_train = df_train["ciphertext"].values
text_test = df_test["ciphertext"].values

tokenizer.fit_on_texts(list(text_train)+list(text_test))

print('Tokenizing train...')
tokenized_text_train = tokenizer.texts_to_sequences(text_train)
print('Tokenizing test...')
tokenized_text_test = tokenizer.texts_to_sequences(text_test)
#Pad sentences
max_len = 30

print('Padding train...')
padded_text_train = pad_sequences(tokenized_text_train, maxlen=max_len)
print('Padding test...')
padded_text_test = pad_sequences(tokenized_text_test, maxlen=max_len)
y = df_train['target']

dummy_y = np_utils.to_categorical(y)
#Split in train and validation
train_x, valid_x, train_y, valid_y = train_test_split(padded_text_train, dummy_y, test_size=0.15, random_state=42)

test_x = np.array(padded_text_test)
#Build LSTM Network model
max_features = 50000

model_lstm = Sequential()
model_lstm.add(Embedding(max_features, 300, input_length=max_len))
model_lstm.add(Bidirectional(CuDNNLSTM(144)))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(96, activation='elu'))
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(20, activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model_lstm.fit(train_x, train_y, validation_split= 0.2, epochs=4, batch_size=64)
#Predict on validation and check F1 score
pred_valid = model_lstm.predict(valid_x)
f1_err = f1_score(np.argmax(valid_y, axis=1), np.argmax(pred_valid, axis=1), average='macro')

print('F1 score on validation set:', f1_err)
#Predict on test set
pred_test = model_lstm.predict(test_x)
final_prediction = np.argmax(pred_test, axis=1)
df_submission['Predicted'] = final_prediction

df_submission.to_csv('submission.csv', index = False)
