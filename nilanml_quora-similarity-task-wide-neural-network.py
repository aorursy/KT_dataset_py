from importlib import reload

import sys

from imp import reload

import warnings

warnings.filterwarnings('ignore')

if sys.version[0] == '2':

    reload(sys)

    sys.setdefaultencoding("utf-8")
import pandas as pd

df = pd.read_csv('../input/questions.csv')

df = df.drop(['id','qid1','qid2'],axis=1)

df.head()
import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

    text = [lemmatizer.lemmatize(token, "v") for token in text]

#     text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text
df['question1'] = df.question1.apply(lambda x: clean_text(str(x)))

df.head()
df['question2'] = df.question2.apply(lambda x: clean_text(str(x)))

df.head()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , CuDNNLSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D, concatenate, dot

from keras.models import Model, Sequential
total_text = pd.concat([df['question1'], df['question2']]).reset_index(drop=True)

max_features = 6000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(total_text)

question_1_sequenced = tokenizer.texts_to_sequences(df['question1'])

question_2_sequenced = tokenizer.texts_to_sequences(df['question2'])
maxlen = 100

question_1_padded = pad_sequences(question_1_sequenced, maxlen=maxlen)

question_2_padded = pad_sequences(question_2_sequenced, maxlen=maxlen)
y = df['is_duplicate']
embedding_size = 128



inp1 = Input(shape=(100,))

inp2 = Input(shape=(100,))



x1 = Embedding(max_features, embedding_size)(inp1)

x2 = Embedding(max_features, embedding_size)(inp2)



x3 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x1)

x4 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x2)



x5 = GlobalMaxPool1D()(x3)

x6 = GlobalMaxPool1D()(x4)



x7 =  dot([x5, x6], axes=1)



x8 = Dense(40, activation='relu')(x7)

x9 = Dropout(0.05)(x8)

x10 = Dense(10, activation='relu')(x9)

output = Dense(1, activation="sigmoid")(x10)



model = Model(inputs=[inp1, inp2], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100

epochs = 3

model.fit([question_1_padded, question_2_padded], y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
