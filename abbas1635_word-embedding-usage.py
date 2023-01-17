import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from textblob import TextBlob

import nltk

import re

from bs4 import BeautifulSoup

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from scipy.sparse import coo_matrix, hstack

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



from keras.models import Sequential

from keras.layers import LSTM

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
#loading the data

data =pd.read_csv("../input/mondaq_data_adeel.csv", header = None)
columns = ['article_id', 'knt', 'unknt', 'her_knt', 'unher_knt', 'company_id', 

           'company_name', 'country_id', 'country_desc', 'primary_topic_id', 'topic_desc', 

           'article_start_date', 'daysold', 'topics', 'coauthors', 'linkedinphoto', 'title', 'body']
data.head()
data.columns = columns
data.head()
plt.figure(figsize=(6,4))

ax = sns.countplot(x="country_desc", data=data)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
#Using BeautifulSoup to clean html tags

data['clean_article_text'] = data['body'].apply(lambda x:BeautifulSoup(x, 'lxml').text)
data.head()
#combine titlt with body

data['entire_text'] = data['title'] + " " + data['clean_article_text']



#remove special characters present in the text

data['entire_text'] = data['entire_text'].apply(lambda x: re.sub(r'[^a-zA-z0-9\s]', ' ',x))
data.head()
data['length_of_text'] = data['entire_text'].apply(lambda x:len(x.split()))
data.head()
data['length_of_text'].describe()
plt.hist(data['length_of_text'], bins=1000);
data['knt'].describe()
data['Reg_clicks_categorical'] = data['knt'].apply(lambda x:  0 if x < 10 else (1 if x <= 22 else (2 if x <= 56 else 3))) 
data['Reg_clicks_categorical'].value_counts()
train_df, val_df = train_test_split(data, test_size = 0.2, random_state= 0)
train_df.shape, val_df.shape
## some config values 

embed_size = 100 # how big is each word vector

max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 1400 # max number of words in a question to use
## fill up the missing values

train_X = train_df["entire_text"].fillna("##").values

val_X = val_df["entire_text"].fillna("##").values



print("before tokenization")

print(train_X.shape)

print(val_X.shape)
## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(data['entire_text'].fillna("##").values))



train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)



print("after tokenization")

print(len(train_X))

print(len(val_X))
## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)
## Get the target values

train_y = train_df['Reg_clicks_categorical'].values

val_y = val_df['Reg_clicks_categorical'].values
train_y = to_categorical(train_y, num_classes=4)

val_y = to_categorical(val_y, num_classes=4)
model = Sequential() 

model.add(Embedding(max_features, embed_size, input_length=maxlen)) 

model.add(LSTM(256)) 

model.add(Dense(4, activation='softmax')) 

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 

print(model.summary()) 
#inp = Input(shape=(maxlen,))

#x = Embedding(max_features, embed_size)(inp)

#x = LSTM(512, return_sequences=True)(x)

#x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

#x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

#x = Flatten()(x)

#x = Dense(64, activation="relu")(x)

#x = Dense(64, activation="relu")(x)

#x = Dense(4, activation="softmax")(x)

#model = Model(inputs=inp, outputs=x)

#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

#model.summary()
model.fit(train_X, train_y,epochs=2, batch_size = 256, validation_data=(val_X, val_y))
y_prediction = model.predict(train_X, batch_size=512, verbose=1)
predictions = []

for i in range(len(y_prediction)):

    predictions.append(np.argmax(y_prediction[i]))
train_df['prediction'] = predictions
confusion_matrix(train_df['Reg_clicks_categorical'], train_df['prediction'])
accuracy_score(train_df['Reg_clicks_categorical'], train_df['prediction'])