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
data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv',usecols=['id','text','target'])

data.head()
from sklearn import preprocessing

from keras.layers import Input,Dense,Embedding,LSTM,Dropout,Activation

from keras.layers import Bidirectional,GlobalMaxPool1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
data.text.values
embedded_size = 100

max_features = 10000

maxlen = 100
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_X = test.text.values
from sklearn.model_selection import train_test_split

train_df ,val_df = train_test_split(data,test_size = 0.1 , random_state = 43)

train_X = train_df.text.values

val_X = val_df.text.values
tokenizer = Tokenizer(num_words = max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)
train_y = train_df.target.values

val_y = val_df.target.values
from keras.models import Model



inp = Input(shape = (maxlen,))

x = Embedding(max_features,embedded_size)(inp)

x = Bidirectional(LSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16,activation='relu')(x)

x = Dropout(0.1)(x)

x = Dense(1,activation = 'sigmoid')(x)

model = Model(inputs = inp,outputs = x)

model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])



print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=10, validation_data=(val_X, val_y))
preds = model.predict([test_X],batch_size = 1024,verbose = 1)
predictions = (preds > 0.5).astype(int)

predictions = np.ndarray.flatten(predictions)
sub2= pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sub2['target'] = predictions

sub2.to_csv("submission3.csv", index=False)

sub2.head()