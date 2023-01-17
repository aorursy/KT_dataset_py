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
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)
plt.style.use("ggplot")

import tensorflow as tf
print('Tensorflow version:', tf.__version__)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.utils as ku 
import numpy as np 
df=pd.read_csv("/kaggle/input/news-summary/news_summary_more.csv")
df.head(10)
df['text'][0]
df['headlines'][0]
headlines=[]
for i in df['headlines']:
    headlines.append(i)
# headlines
len(headlines)
tokenizer=Tokenizer(num_words=10000)
tokenizer.fit_on_texts(headlines[:500])
total_words=len(tokenizer.word_index)+1
total_words
sequences=[]
# headlines[:500]
len(headlines)
for l in headlines[:5000]:
     token = tokenizer.texts_to_sequences([l])[0]
#      print(token)
     for i in range(1,len(token)):
       ngrams_seq=token[:i+1]
       sequences.append(ngrams_seq)
# sequences
len(sequences)
maxl=0
for i in sequences:
    k=len(i)
    if k>maxl:
        maxl=k
maxl
data= pad_sequences(sequences, maxlen=maxl)
data
data.shape
predictors=data[:,:-1]
predictors
predictors.shape
labels=data[:,-1]
labels
labels.shape
labels=ku.to_categorical(labels,num_classes=total_words)
labels
labels.shape
model = Sequential()
model.add(Embedding(input_dim=total_words,output_dim=80,input_length=15))#input length is 15 not 16 as we have taken the last column for labels for 16-1=15
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=150,return_sequences=False)))#if return sequences is false,then it will return a 2-D array,if true then it will return a 3-D array..
model.add(Dense(total_words,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(predictors, labels, epochs=100, verbose=1)
accuracy = history.history['accuracy']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.title('Training accuracy')
loss = history.history['loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training accuracy')
plt.title('Training accuracy')
output_word = ""
test_text = "President Donald Trump"
next_words = 5

for num in range(next_words):
	token = tokenizer.texts_to_sequences([test_text])
	new_pad = pad_sequences(token, maxlen=15)
	predicted = model.predict_classes(new_pad, verbose=0)
	
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	test_text += " " + output_word
print(test_text)
test_text = "India and China"
next_words = 5

for num in range(next_words):
	token = tokenizer.texts_to_sequences([test_text])
	new_pad = pad_sequences(token, maxlen=15)
	predicted = model.predict_classes(new_pad, verbose=0)
	
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	test_text += " " + output_word
print(test_text)
test_text = "BCCI"
next_words = 5

for num in range(next_words):
	token = tokenizer.texts_to_sequences([test_text])
	new_pad = pad_sequences(token, maxlen=15)
	predicted = model.predict_classes(new_pad, verbose=0)
	
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	test_text += " " + output_word
print(test_text)