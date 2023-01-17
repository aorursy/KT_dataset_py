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
df=pd.read_csv("/kaggle/input/movie-review/movie_review.csv")
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,GlobalAveragePooling1D,Flatten,Embedding,Conv1D,MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

df.head()
df=df[['cv_tag','text','tag']]
df=df.groupby(['cv_tag', 'tag'], as_index = False).agg({'text': ' '.join})
df=df[['text','tag']]
df.head()
x,y=df['text'],df['tag']
train_x,test_x=x[:1600],x[1600:]
train_y,test_y=y[:1600],y[1600:]

train_labels={'pos':1,'neg':0}
train_y=train_y.map(train_labels)
test_y=test_y.map(train_labels)
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
l=stopwords.words('english')
def clean(text):
    text = text.lower()
    text = re.sub('\[.*?\,]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    
    
    return text
train_x=train_x.apply(lambda x:clean(x))
test_x=test_x.apply(lambda x:clean(x))
vocab_size=39000
dimention=100
oop_token="<oop>"
max_length=2500
tokenizer=Tokenizer(num_words=vocab_size,oov_token=oop_token)
tokenizer.fit_on_texts(train_x)
sequences=tokenizer.texts_to_sequences(train_x)
padded=pad_sequences(sequences,maxlen=max_length)
test_sequences=tokenizer.texts_to_sequences(test_x)
padded_test=pad_sequences(test_sequences,maxlen=max_length)

model=Sequential()
model.add(Embedding(vocab_size,dimention,input_length=max_length))
model.add(Conv1D(32, 8, activation="relu"))
model.add(MaxPooling1D(2))
model.add(GlobalAveragePooling1D())
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(
   padded,train_y,epochs=5,validation_data=(padded_test,test_y)

)
