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
import numpy as np 
import pandas as pd 
import os

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import re

from nltk.corpus import stopwords
import itertools
import nltk
nltk.download('stopwords')
stops = set(stopwords.words('english'))

def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text
df_original = pd.read_csv("/kaggle/input/nnfl-lab-3-nlp/nlp_train.csv", sep=",")
df = pd.DataFrame.copy(df_original)
df.head()
test_df = pd.read_csv("/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv", sep=",")
submission_df = test_df.copy()
test_df.info()
test_df = test_df.drop(['offensive_language'], axis = 1)
nb_samples = test_df.shape[0]
nb_samples
max_seq_length = 100
vocabulary = dict()
inverse_vocabulary = ['<unk>']

# Iterate over the questions only of both training and test datasets
for dataset in [df,test_df]:
    for index, row in dataset.iterrows():

        # Iterate through the text of both questions of the row
        for sentence in ['tweet']:

            s2n = []  # q2n -> sentence numbers representation
            for word in text_to_word_list(row[sentence]):

                # Check for unwanted words
                if word in stops:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    s2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    s2n.append(vocabulary[word])

            # Replace questions as word to question as number representation
            dataset.at[index, sentence] = s2n
embeddings_index = {}
f = open('/kaggle/input/glovetwitter100d/glove.twitter.27B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
EMBEDDING_DIM = 100
embeddings = np.zeros((len(vocabulary) + 1, EMBEDDING_DIM))
for word, i in vocabulary.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embeddings[i] = embedding_vector
from keras.layers import Embedding

embedding_layer = Embedding(len(embeddings),
                            EMBEDDING_DIM,
                            weights=[embeddings],
                            input_length=max_seq_length,
                            trainable=False)
from keras.preprocessing.sequence import pad_sequences
max_length_of_text = 100

X = pad_sequences(df['tweet'], maxlen=max_seq_length)
X_test = pad_sequences(test_df['tweet'], maxlen=max_seq_length)
y = df['offensive_language']
X_train, X_validation, y_train, y_validation = train_test_split(X,y, test_size = 0.2, random_state = 42)
print(X_train.shape,y_train.shape)
print(X_validation.shape,y_validation.shape)
lstm_out = 128
length_of_text = 100
batch_size = 32
inputs = Input(shape=(max_seq_length,), dtype='int32')
x = embedding_layer(inputs)
x = Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))(x)
x = Dense(1,activation='linear')(x)
model = Model(inputs, x)
print(model.summary())
model.compile(loss='mean_squared_error',optimizer='adam');
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=4)
model.fit(X_train, y_train, batch_size = batch_size, epochs = 10, callbacks=[es])
score = model.evaluate(X_validation, y_validation, batch_size = batch_size)
print("Validation Loss: %.2f" % (score))
model.save_weights("model_try.h5")
predict = model.predict(X_test)
test_df.info()
submission_df['offensive_language'] = predict
submission_df.head()
submission_df.to_csv('submission_try.csv', index=False)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = submission_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)
