import numpy as np
import pandas as pd
import re, sys, os, csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

!ls ../input/
df = pd.read_csv('../input/text_emotion.csv')
df.head()
import re

# Function to clean data ... will be useful later
def post_cleaner(post):
    """cleans individual posts`.
    Args:
        post-string
    Returns:
         cleaned up post`.
    """
    # Covert all uppercase characters to lower case
    post = post.lower() 
    
    # Remove |||
    post=post.replace('|||',"") 

    # Remove URLs, links etc
    post = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', post, flags=re.MULTILINE) 
    # This would have removed most of the links but probably not all 

    # Remove puntuations 
    puncs1=['@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\','"',"'",';',':','<','>','/']
    for punc in puncs1:
        post=post.replace(punc,'') 

    puncs2=[',','.','?','!','\n']
    for punc in puncs2:
        post=post.replace(punc,' ') 
    # Remove extra white spaces
    post=re.sub( '\s+', ' ', post ).strip()
    return post
def vectorise_label(label):
    if label == "empty":return 0
    elif label == "sadness":return 3
    elif label == "enthusiasm":return 1
    elif label == "neutral":return 0
    elif label == "worry":return 3
    elif label == "surprise":return 2
    elif label == "love":return 2
    elif label == "fun":return 1
    elif label == "hate":return 4
    elif label == "happiness":return 1
    elif label == "boredom":return 0
    elif label == "relief":return 1
    elif label == "anger":return 5
preproc = df
preproc['content'] = preproc['content'].apply(lambda x: post_cleaner(x))
preproc['sentiment_id'] = preproc['sentiment'].apply(lambda x: vectorise_label(x))
results = set()
preproc.content.str.split().apply(results.update)
len(results)
length = preproc.content.apply(len)
length.max()
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tok = Tokenizer()
tok.fit_on_texts(preproc.content)
docs = tok.texts_to_sequences(preproc.content)

MAX_LEN = 152
MAX_VOCAB = 40000

padded = pad_sequences(docs, maxlen=MAX_LEN, padding='post')
padded.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(padded, preproc['sentiment_id'], random_state = 0)
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, GRU, Bidirectional, BatchNormalization, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
def create_RNN():
    model = Sequential()
    model.add(Embedding(MAX_VOCAB, 32, input_length=MAX_LEN))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(GRU(units=64, dropout=0.2, recurrent_dropout=0.2))
    model.add(BatchNormalization())
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
    model.add(Dense(16, activation='softmax'))
    model.compile(Adam(0.1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
rnn = create_RNN()
rnn.summary()
!mkdir saved_models
!ls
callbacks = [EarlyStopping(min_delta=0.001, verbose=1), ModelCheckpoint(filepath='saved_models/rnn.weights.best.hdf5', 
                               verbose=1, save_best_only=True)]

epochs = 7
model_info = rnn.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, callbacks=callbacks, batch_size=64)
model_dir = 'saved_models/mbit_rnn_model.json'
weights_dir = 'saved_models/mbit_rnn_weights.h5'

model_json = rnn.to_json()
with open(model_dir, 'w') as file:
    file.write(model_json)

rnn.save_weights(weights_dir)
import matplotlib.pyplot as plt
%matplotlib inline

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    
plot_model_history(model_info)
