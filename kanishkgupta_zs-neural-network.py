# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import re

import nltk

from nltk.corpus import stopwords



from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten

from keras.layers import GlobalMaxPooling1D

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
df=pd.read_csv("/kaggle/input/train_file.csv")

df_test=pd.read_csv("/kaggle/input/test_file.csv")

df.shape
df.head()
# # x1=[df['Title'],df['Headline']]

# # x2=[df['SentimentTitle'],df['SentimentHeadline']]

# concatvalues = np.concatenate([df.Title.values,df.Headline.values])

# df_new = pd.concat([df,pd.DataFrame(concatvalues)], ignore_index=True, axis=1)

# df_new.columns = np.append(df.columns.values, "review")



# df_new.shape
df_new.head()
# concatvalues_sent = np.concatenate([df.SentimentTitle.values,df.SentimentHeadline.values])

# df_new_sent = pd.concat([df,pd.DataFrame(concatvalues_sent)], ignore_index=True, axis=1)

# df_new_sent.columns = np.append(df.columns.values, "Sentiments")



# df_new_sent.shape
# df_new_sent.tail()
# df_train=pd.concat([df_new,df_new_sent],axis=1)

# df_train.shape
def preprocess_text(sen):

    



    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sen)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)



    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)



    return sentence
X_title = []

sentences = list(df['Title'])

for sen in sentences:

    X_title.append(preprocess_text(sen))
X_title
X_title_test = []

sentences = list(df_test['Title'])

for sen in sentences:

    X_title_test.append(preprocess_text(sen))

X_title_test    
tokenizer = Tokenizer()

total=X_title+X_title_test

tokenizer.fit_on_texts(total)
max_length=max([len(s.split()) for s in total])
max_length
vocab_size = len(tokenizer.word_index) + 1
vocab_size
X_title = tokenizer.texts_to_sequences(X_title)

X_title_test = tokenizer.texts_to_sequences(X_title_test)

len(X_title)
X_title = pad_sequences(X_title, padding='post', maxlen=max_length)

X_title_test = pad_sequences(X_title_test, padding='post', maxlen=max_length)

model = Sequential()

embedding_layer = Embedding(vocab_size, 100, input_length=max_length)

model.add(embedding_layer)

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(256,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='tanh'))

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

print(model.summary())

y_title=df.iloc[:,9].values

model.fit(X_title, y_title, batch_size=64, epochs=10, verbose=1)





df.columns
y_pred_title=model.predict(X_title_test)
y_pred_title
def preprocess_text(sen):

    



    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sen)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)



    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)



    return sentence
X_head = []

sentences = list(df['Headline'])

for sen in sentences:

    X_head.append(preprocess_text(sen))
X_head_test = []

sentences = list(df_test['Headline'])

for sen in sentences:

    X_head_test.append(preprocess_text(sen))

X_head_test    


total_head=X_head+X_head_test

tokenizer.fit_on_texts(total_head)
max_length_head=max([len(s.split()) for s in total_head])
max_length_head
vocab_size = len(tokenizer.word_index) + 1
vocab_size
X_head = tokenizer.texts_to_sequences(X_head)

X_head_test = tokenizer.texts_to_sequences(X_head_test)

X_head = pad_sequences(X_head, padding='post', maxlen=max_length_head)

X_head_test = pad_sequences(X_head_test, padding='post', maxlen=max_length_head)
# X_head.shape
model = Sequential()

embedding_layer = Embedding(vocab_size, 100, input_length=max_length_head)

model.add(embedding_layer)

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='tanh'))

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

print(model.summary())
y_head=df.iloc[:,10].values

model.fit(X_head, y_head, batch_size=64, epochs=10, verbose=1)

y_head.shape
y_pred_head=model.predict(X_head_test)
np.savetxt("nn.csv", y_pred_title,fmt="%f")

np.savetxt("nn_head.csv", y_pred_head,fmt="%f")