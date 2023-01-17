import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator#
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from nltk.corpus import stopwords
from keras.utils import to_categorical
import os
print(os.listdir("../input"))
train_df = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t")
test_df= pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t")
pos_df = pd.read_csv("../input/pos-neg-files/positive words.txt", sep="\n", header=None)
neg_df = pd.read_csv("../input/pos-neg-files/Negative words.txt", sep="\n", header=None, encoding = "ISO-8859-1")
pos_df.columns = ['words']
neg_df.columns = ['words']
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize, sent_tokenize
import string
for df in [train_df, test_df]:
    df['words_length'] = df['Phrase'].apply(lambda x: len(x))
    df['sent_length'] = df['Phrase'].apply(lambda x: len(word_tokenize(x)))
    df['no_stops'] = df['Phrase'].apply(lambda x: len([w for w in word_tokenize(x.lower()) if w in stop_words]))
    df['no_non_stops'] = df['Phrase'].apply(lambda x: len([w for w in word_tokenize(x.lower()) if w not in stop_words]))
    df['no_punctuations'] = df['Phrase'].apply(lambda x: 
                                               len([w for w in word_tokenize(x.lower()) if w in string.punctuation if w not in "." 
                                                   if w not in ","]))
    
    df['pos_words'] = df['Phrase'].apply(lambda x: len([w for w in word_tokenize(x.lower()) if w in pos_df.words.values]))
    df['neg_words'] = df['Phrase'].apply(lambda x: len([w for w in word_tokenize(x.lower()) if w in neg_df.words.values]))
    df['neutral_words'] = df['Phrase'].apply(lambda x: len([w for w in word_tokenize(x.lower()) if w not in neg_df.words.values
                                                           if w not in pos_df.words.values]))
train_df['Phrase'][(train_df['words_length']==1) & (train_df['Sentiment']==0) ] = "bad" 
train_df['Phrase'][(train_df['words_length']==1)& (train_df['Sentiment']==1 )] = "bad" 
train_df['Phrase'][(train_df['words_length']==1) & (train_df['Sentiment']==2) ] = "seem"
train_df['Phrase'][(train_df['words_length']==2) & (train_df['Sentiment'] >=2) ] = "seem"
test_df['Phrase'][(test_df['words_length']==1)]  = "seem"
test_df['Phrase'][(test_df['words_length']==2) & ((test_df['Phrase'] != "no") | (test_df['Phrase'] != "No"))] = "seem"
#new words length
for df in [train_df, test_df]:
    df['words_length'] = df['Phrase'].apply(lambda x: len(x))
train_df.head()
test_df.head()
old_train = train_df[['words_length','sent_length', 'no_stops', 'no_non_stops', 'no_punctuations',
                   'pos_words', 'neg_words', 'neutral_words']]
old_test = test_df[['words_length','sent_length', 'no_stops', 'no_non_stops', 'no_punctuations',
                   'pos_words', 'neg_words', 'neutral_words']]
y_train = train_df['Sentiment']
old_train =(old_train - old_train.min())/(old_train.max() - old_train.min())
old_test =(old_test - old_test.min())/(old_test.max() - old_test.min())
old_train.head()
load_train_df = train_df.copy()
load_test_df = test_df.copy()
#split the data for training and cross validation
train_df, val_df = train_test_split(train_df, test_size = 0.1, random_state= 144)
print(train_df.shape)
print(val_df.shape)
## some config values 
embed_size = 100 # how big is each word vector
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

## fill up the missing values
train_X = train_df["Phrase"].fillna("##").values
val_X = val_df["Phrase"].fillna("##").values
test_X = test_df['Phrase'].fillna("##").values
print("before tokenization")
print(train_X.shape)
print(val_X.shape)
print(test_X.shape)

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

print("after tokenization")
print(len(train_X))
print(len(val_X))
print(len(test_X))
## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

print("after padding")
print(len(train_X))
print(len(val_X))
print(len(test_X))

## Get the target values
train_y = train_df['Sentiment'].values
val_y = val_df['Sentiment'].values
#shuffling the data
np.random.seed(2018)
trn_idx = np.random.permutation(len(train_X))
val_idx = np.random.permutation(len(val_X))

train_y = train_df['Sentiment'].values
val_y = val_df['Sentiment'].values

train_X = train_X[trn_idx]
val_X = val_X[val_idx]
train_y = train_y[trn_idx]
val_y = val_y[val_idx]
new_train_x = old_train.loc[trn_idx].values
new_val_X = old_train.loc[val_idx].values
#new_train_x = pad_sequences(new_train_x, maxlen=maxlen)
#new_val_X = pad_sequences(new_val_X, maxlen=maxlen)
print(new_train_x.shape)
print(new_val_X.shape)
final_train_x = pd.concat([pd.DataFrame(train_X), pd.DataFrame(new_train_x)], axis=1)
final_val_x  =  pd.concat([pd.DataFrame(val_X), pd.DataFrame(new_val_X)], axis=1)
final_test_x = pd.concat([pd.DataFrame(test_X), pd.DataFrame(old_test.values)], axis=1)
maxlen = 60
final_train_x = pad_sequences(final_train_x.values, maxlen=maxlen)
final_val_x = pad_sequences(final_val_x.values, maxlen=maxlen)
final_test_x = pad_sequences(final_test_x.values, maxlen=maxlen)
print(final_train_x.shape)
print(final_val_x.shape)
print(final_test_x.shape)
#other_inp = Input(shape=(8,))
#other_inp = Embedding(8, 8)(other_inp)
#other_inp = Bidirectional(CuDNNLSTM(128, return_sequences=True))(other_inp)
#other_inp = Flatten()(other_inp)
#other_inp = Dense(64, activation="relu")(other_inp)
#auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(other_inp)
#x = Dense(5, activation="softmax")(x)
#model = Model(inputs=other_inp, outputs=x)
#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
from tensorflow import keras
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(5, activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
model.summary()
train_y = to_categorical(train_y, num_classes=5)
val_y = to_categorical(val_y, num_classes=5)
# ## Train the model 
model.fit(final_train_x, train_y, batch_size=512, epochs=4, validation_data=(final_val_x, val_y))
pred_glove_val_y = model.predict([final_test_x], batch_size=1024, verbose=1)
predictions = []
for i in range(len(pred_glove_val_y)):
    predictions.append(np.argmax(pred_glove_val_y[i]))
len(predictions)
submission_df = pd.DataFrame()
submission_df['PhraseId'] = test_df['PhraseId']
submission_df['Sentiment'] = predictions 
submission_df.to_csv("submission.csv", index=False)
submission_df.head()
submission_df['Sentiment'].value_counts()
