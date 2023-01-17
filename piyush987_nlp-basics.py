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
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, MaxPooling1D, Convolution1D, Dropout, Activation, GlobalMaxPool1D
from keras.models import Model, Sequential
from tensorflow.keras import regularizers
from gensim.models import Word2Vec, Phrases
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df
df.info() #All non NULL values in both columns
df.sentiment.value_counts() #Sentiments need to be converted to 1 for positive, 0 for -ve
df.iloc[0][0] #Data needs to be cleaned as it contains some html tags(<br /><br />), uppercase letters
def clean_data(text):
  text = re.sub(r'<br />', ' ', text) #Removes Html tag
  text = re.sub(r'[^\ a-zA-Z0-9]+', '', text)  #Removes non alphanumeric
  text = re.sub(r'^\s*|\s\s*', ' ', text).strip() #Removes extra whitespace, tabs
  stop_words = set(stopwords.words('english')) 
  lemmatizer = WordNetLemmatizer()
  text = text.lower().split() #Converts text to lowercase
  cleaned_text = list()
  for word in text:        
    if word in stop_words:    #Removes Stopwords, i.e words that don't convey any meaningful context/sentiments
      continue    
    word = lemmatizer.lemmatize(word, pos = 'v')    #Lemmatize words, pos = verbs, i.e playing, played becomes play
    cleaned_text.append(word)
  text = ' '.join(cleaned_text)
  return text

df['cleaned_review'] = df['review'].apply(lambda x: clean_data(x))
df
def convert_sentiment_to_int(text):  #Convert sentiment positive to 1, negative to 0
  if(text.lower() == 'positive'):
    text = 1
  else:
    text = 0
  return text

df['sentiment'] = df['sentiment'].apply(lambda x: convert_sentiment_to_int(x))
df
result = [len(x) for x in [df['cleaned_review'].iloc[i].split() for i in range(50000)]]
np.mean(result) #Mean no of words in each cleaned review
X_train = [text for text in list(df['cleaned_review'].iloc[:25000])] #Preparation of X,Y
X_test = [text for text in list(df['cleaned_review'].iloc[25000:])]
Y_train = [text for text in list(df['sentiment'].iloc[:25000])]
Y_test = [text for text in list(df['sentiment'].iloc[25000:])]
print(len(np.unique(np.hstack(X_train)))) #No of unique words in cleaned review
X = [text for text in list(df['cleaned_review'])] 
max_vocab = 10000  #Max features
max_sent_length = 150  #Max word length of every review
tokenizer = Tokenizer(num_words = max_vocab)
tokenizer.fit_on_texts(X)
X_train_tokenized = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen = max_sent_length) #Tokenization, i.e converting words to int
X_test_tokenized = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen = max_sent_length)
def lr_scheduler(epoch, lr):      #For tuning the learning rate
    if epoch > 0:
        lr = 0.0001
        return lr
    return lr
model = Sequential()  #Sequential layers
model.add(Embedding(max_vocab, 150, input_length = max_sent_length)) #Embedding layer
model.add(Bidirectional(LSTM(60, return_sequences = True, dropout = 0.2))) #BiLSTM
model.add(Convolution1D(32, 3, padding = 'valid', activation = 'relu'))  # 1D Conv
model.add(GlobalMaxPool1D())
model.add(Dropout(0.6))     #High droput to reduce overfitting
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation =  'sigmoid'))
optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])  #Adam gave better results than SGD
print(model.summary())
batch_size = 64
epochs = 10
callbacks = [keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]
hist = model.fit(X_train_tokenized, np.array(Y_train), batch_size = batch_size, epochs = epochs, verbose = 1,  validation_data = (X_test_tokenized, np.array(Y_test)),  callbacks = callbacks)
model.evaluate(X_test_tokenized, np.array(Y_test))
y_train_pred = model.predict_classes(X_train_tokenized)    #Predicted output
y_test_pred = model.predict_classes(X_test_tokenized)
confusion_matrix(Y_train,y_train_pred)
loss_train = hist.history['loss']
loss_val = hist.history['val_loss']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()      #The below graph shows that the model is a bit overfit :(
loss_train = hist.history['accuracy']
loss_val = hist.history['val_accuracy']
epochs = range(1,11)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
