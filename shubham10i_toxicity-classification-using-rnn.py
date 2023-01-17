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
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
train
train = train[['target','comment_text']]
train.head()
x = train['comment_text'].tolist()
x[:5]
y = train['target'].tolist()
y
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer  
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, GRU, Dropout,Embedding ,Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
vocab_sz = 10000
maxlen=100
x_train, x_test, y_train, y_test = train_test_split(x[:60000], y[:60000], test_size=0.3, random_state=42)
tok = Tokenizer(num_words=vocab_sz, oov_token='UNK')
tok.fit_on_texts(x)
x_train = tok.texts_to_sequences(x_train)
x_test = tok.texts_to_sequences(x_test )
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
model = Sequential()
model.add(Embedding(vocab_sz+1, 50, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.summary()
y_train= np.array (y_train)
y_test= np.array(y_test)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=64, epochs=10
          ,validation_data=(x_test, y_test))
test_df= pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
test_df= test_df['comment_text']
test_df.head()
xt = test_df.tolist()
xt[:5]
vocab_sz = 10000
maxlen=100
tok = Tokenizer(num_words=vocab_sz, oov_token='UNK')
tok.fit_on_texts(xt)
xt = tok.texts_to_sequences(xt)
xt = pad_sequences(xt, maxlen=maxlen)
y_pred= model.predict(xt)
y_pred = np.where(y_pred>=0.5,1.,0.)
sub_df = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
sub_df['prediction'] = y_pred
sub_df.to_csv('submission.csv',index = False)
