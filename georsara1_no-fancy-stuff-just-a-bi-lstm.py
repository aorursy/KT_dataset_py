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
#Import data
print('Importing data...')
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

print("Train shape : ",df_train.shape)
print("Test shape : ",df_test.shape)
#Shuffle data
df_train = df_train.sample(frac=1).reset_index(drop=True)
max_features = 80000
max_len = 40
#Tokenize sentences
tokenizer = Tokenizer(num_words=max_features, filters='#$%&()*+,-./:;<=>@[\]^_`{|}~',)

text_train = df_train["question_text"].fillna("_na_").values
text_test = df_test["question_text"].fillna("_na_").values

tokenizer.fit_on_texts(list(text_train)+list(text_test))

print('Tokenizing train...')
tokenized_text_train = tokenizer.texts_to_sequences(text_train)
print('Tokenizing test...')
tokenized_text_test = tokenizer.texts_to_sequences(text_test)
#Pad sentences
print('Padding train...')
tokenized_text_train = pad_sequences(tokenized_text_train, maxlen=max_len)
print('Padding test...')
tokenized_text_test = pad_sequences(tokenized_text_test, maxlen=max_len)
#Split in train and validation
train_x, valid_x, train_y, valid_y = train_test_split(tokenized_text_train, df_train['target'], test_size=0.15, random_state=42)

test_x = np.array(tokenized_text_test)
#Build LSTM Network model
model_lstm = Sequential()
model_lstm.add(Embedding(max_features, 96, input_length=max_len))
model_lstm.add(Bidirectional(CuDNNLSTM(96)))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(4, activation='elu'))
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])


# Fit the model
model_lstm.fit(train_x, train_y, validation_split= 0.2, epochs=2, batch_size=256)
# Evaluation of the model on validation set
valid_pred_prob_lstm = model_lstm.predict(valid_x)
valid_pred_01_lstm = np.where(valid_pred_prob_lstm>0.27,1,0)
valid_pred_01_lstm = [int(item) for item in valid_pred_01_lstm]
#Calculate F1 score on validation data
f1_quora  = f1_score(valid_y, valid_pred_01_lstm)
print('Validation F1 score LSTM:', f1_quora)
# Final evaluation of the ensemble
test_pred_prob_lstm = model_lstm.predict(test_x)
test_pred_01_lstm = np.where(test_pred_prob_lstm>0.27,1,0)
test_pred_01_lstm = [int(item) for item in test_pred_01_lstm]
#Write to dataframe
submit_df = pd.DataFrame({'qid':df_test['qid'].values, 'prediction': test_pred_01_lstm})
submit_df.to_csv("submission.csv", index=False)