# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
movie_reviews = pd.read_csv("/kaggle/input/IMDB Dataset.csv")

movie_reviews.isnull().values.any()

movie_reviews.shape
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)
X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))
y = movie_reviews['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

movie_reviews.head()
movie_reviews["review"][10]
import seaborn as sns

sns.countplot(x='sentiment', data=movie_reviews)
#configuration  parameters
LATENT_DIM_DECODER = 400
BATCH_SIZE =128
EPOCHS = 20
LATENT_DIM = 400
NUM_SAMPLES = 50000
MAX_SEQUENCE_LEN = 1000
MAX_NUM_WORDS = 50000
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1
#sequence_train = tokenizer.texts_to_sequences(train_final)
#sequence_test = tokenizer2.texts_to_sequences(test_final)
# get the word to index mapping for input language
#word2idx_inputs = tokenizer.word_index
#print('Found %s unique input tokens.' % len(word2idx_inputs))
#max_len = [len(s) for s in sequence_train]
#print(max(max_len))
#dimension of input to the layer should be constant
#scaling each comment sequence to a fixed length to 200
#comments smaller than 200 will be padded with zeros to make their length as 200
max_len=800
#pad the train and text sequence to be of fixed length (in keras input in lstm should be of fixed length sequnece)
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
from keras.layers import Input,SpatialDropout1D,Conv1D,LSTM,GlobalMaxPool1D,Embedding,Dropout,Bidirectional,GlobalMaxPool1D
from keras.models import Sequential
model = Sequential()
#input_ = Input(shape=(max_len,))
embed_layer = Embedding(vocab_size,300,input_length = 800,mask_zero = True)
model.add(embed_layer)
model.add(SpatialDropout1D(0.4))
cnn = Conv1D(128,3)
LSTM_layer =LSTM(64, return_sequences = True,name='rnn_layer',recurrent_dropout = 0.4)
#LSTM_layer =LSTM(128, return_sequences = True,name='rnn_layer')
#sec_LSTM_layer = Bidirectional(LSTM(256, return_sequences=True, name='BI2_lstm_layer'))(LSTM_layer)
model.add(cnn)
model.add(LSTM_layer)

model.add( Dropout(0.55))
#dimension reduction using pooling layer
from keras.layers import GlobalAvgPool1D
model.add(GlobalMaxPool1D())
model.add(Dropout(0.55))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.55))
model.add(Dense(1,activation='sigmoid'))
from keras.optimizers import Adam
model.compile(loss = 'binary_crossentropy',
             optimizer = Adam(lr=0.001),
             metrics = ['accuracy'])
from keras.models import Model
from keras.optimizers import Adagrad,Adam,RMSprop,Adamax
#model = Model(inputs=input_ , outputs = output_dense)
history = model.fit(X_train, y_train, batch_size=128, epochs=3, verbose=1, validation_split=0.3)


model.summary()
model.save('/kaggle/working/bin_sentiment_try3.h5')
import simplejson as json
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
from keras.models import load_model
model2 = load_model('/kaggle/working/bin_sentiment_try3.h5')
model2.summary()
from keras.preprocessing.text import tokenizer_from_json
import simplejson as json
with open('/kaggle/working/tokenizer.json') as f:
    data = json.load(f)
    tokenizer_load = tokenizer_from_json(data)
from keras.preprocessing.sequence import pad_sequences
s = ['i did not liked it, it was not good']
sequences_custom = tokenizer_load.texts_to_sequences(s)
padded_seq_cus = pad_sequences(sequences_custom,maxlen=1000,padding  = 'post')
x = model2.predict(padded_seq_cus)
print(x)
if x >0.5:
    print("positive")
else:
    print("negative")
print(padded_seq_cus.shape)
print(type(padded_seq_cus))
