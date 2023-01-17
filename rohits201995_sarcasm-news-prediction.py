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
df=pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json",lines=True)
df.head()
del df['article_link'] 
df.head()
import missingno as msno
msno.matrix(df)
import seaborn as sns
sns.countplot(df['is_sarcastic'])
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('stopwords')
corpus = []
for i in range(0, len(df)):
    text = re.sub('[^a-zA-Z]', ' ', df['headline'][i])
    text = text.lower()
    text = text.split()
    
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
corpus
words=[]
for word in corpus:
    words.append(word.split())
words
    
leng=[]
for word in words:
    leng.append(len(word))
print(max(leng)) 
from tensorflow.keras.preprocessing.text import one_hot
sent=[]
for i in range(len(df)):
    sent.append(df['headline'][i])
voc_size=10000
onehot_repr=[one_hot(words,voc_size)for words in sent] 
print(onehot_repr)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dropout
sent_len=40
embedded_sent=pad_sequences(onehot_repr,padding='pre',maxlen=sent_len)
print(embedded_sent)
from tensorflow.keras import layers
from keras.layers import Bidirectional
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
dim=20
model=Sequential()
model.add(Embedding(voc_size,10,input_length=sent_len))
model.add(Bidirectional(LSTM(128, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(40, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 5
history = model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
Y=df['is_sarcastic']
X=pd.DataFrame(embedded_sent)
Y.head()