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
import tensorflow as tf

print(tf.test.gpu_device_name())

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True
import tensorflow as tf



import pandas as pd



import matplotlib.pyplot as plt



from flask import Flask,render_template,url_for,request



import numpy as np



# import warnings



# warnings.filterwarnings('ignore')



from tensorflow.keras.preprocessing.text import Tokenizer



from tensorflow.keras.preprocessing.sequence import pad_sequences



from tensorflow.keras.models import Sequential



from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D, Flatten









file = pd.read_csv('/kaggle/input/cvcsdf/spam.csv', encoding="latin-1")

file.head()

cols = [0, 1]





file['label'] = file['v1'].map({'ham': 0, 'spam': 1})



file_t = file[file.columns[cols]]



train_file = file_t[0:5000]



train_label = file['label'][0:5000]



test_file = file_t[5000:5560]



test_label = file['label'][5000:5560]



tokenizer = Tokenizer(num_words = 400, split = (' '))



tokenizer.fit_on_texts(train_file['v2'].values)



X = tokenizer.texts_to_sequences(train_file['v2'].values)



x = pad_sequences(X, maxlen = 400)



# tokenizer.fit_on_texts(file_t['v2'].values)



Y = tokenizer.texts_to_sequences(test_file['v2'].values)



y = pad_sequences(Y, maxlen = 400)

# x = x.reshape(-1,400,1)





train_labels = np.asarray(train_label)

test_labels = np.asarray(test_label)
model = Sequential()

model.add(Embedding(10000, 400, input_length = 400))

model.add(SpatialDropout1D(rate = 0.2))

model.add(LSTM(64, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True))

model.add(LSTM(64, dropout = 0.2, recurrent_dropout = 0.2))

model.add(Flatten())

model.add(Dense(32,activation = 'relu'))

model.add(Dense(4,activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

with tf.device('/gpu:0'):

    model.fit(x,train_labels, epochs = 2, batch_size = 64,validation_data = (y, test_labels))
# import gc

# gc.collect()
# data = file_t['v2'][2]

# df= r'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C\'s apply 08452810075over18\'s'

# file2 = pd.read_csv('/kaggle/input/spam-ham/spam.csv', encoding="latin-1")

# file22 = file2[file2.columns[cols]]

#Y1 = tokenizer.texts_to_sequences(file['v2'][5566:5567])

# Y1 = tokenizer.texts_to_sequences(y1)

# y1 = pad_sequences(Y1, maxlen = 400)

ds = "hello have you seen and discussed this article and his approach thank you URL hell there are no rules here we re trying to accomplish something thomas alva edison this URL email is sponsored by osdn t..."

Y1 = tokenizer.texts_to_sequences([ds])

y1 = pad_sequences(Y1, maxlen = 400)
file.shape
g = model.predict_classes(y1)
if g==0 : 

    print("ham")

else:

    print("spam")