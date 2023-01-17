from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split as TTS
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")
sample_df = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
train_df.head()
X_train = train_df["text"].values
X_val = test_df["text"].values
y_train = train_df["target"].values
# y_val = test_df["target"].values
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\[@,;]]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words("english"))

def process(text):
    
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub('',text)
    text = BAD_SYMBOLS_RE.sub('',text)
    
    text = ' '.join([x for x in text.split() if x and x not in STOPWORDS])
    return text
X_train = [process(x) for x in X_train]
X_train = np.array(X_train)
vocab_size = 1000
trunc='post'
max_length = 150
tokenizer = Tokenizer(num_words=vocab_size,oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(X_train)
padded = pad_sequences(sequences,maxlen=max_length,truncating=trunc,padding='post')

val_sequences = tokenizer.texts_to_sequences(X_val)
val_padded = pad_sequences(val_sequences,maxlen=max_length,truncating=trunc,padding='post')
from keras.models import Sequential
from keras.layers import Embedding, Dense, GlobalAveragePooling1D, Bidirectional, LSTM, Conv1D, Dropout, Activation
model = Sequential()

model.add(Embedding(vocab_size,20,input_length=max_length))
model.add(Bidirectional(LSTM(64)))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(padded,y_train,epochs=20,verbose=1)
def show_history(history):
    fig,ax = plt.subplots(1,2,figsize=(15,5))
    
    ax[0].plot(history.history['loss'])
    ax[1].plot(history.history['accuracy'])
    
    ax[0].set_title("Loss")
    ax[0].set_title("Accuracy")
    pass
show_history(history)
def predict_and_save(test,model,name):
    
    test = [process(x) for x in test]
    test = np.array(test)
    test_sequences = tokenizer.texts_to_sequences(test)
    test_padded = pad_sequences(test_sequences,maxlen=max_length,truncating=trunc,padding='post')
    
    pred = model.predict(test_padded)
    pred = np.argmax(pred,axis=1)
    pred = np.array(pred)
    sample_df.target = pred
    sample_df.to_csv(name,index=False)
    pass
predict_and_save(X_val,model,"model_lstm.csv")