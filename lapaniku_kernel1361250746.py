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
df = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
df.head()
df = df.fillna('')
df['all_text'] = df.apply(lambda x: ' '.join([x['title'], x['location'], x['department'], 
                                             x['company_profile'], x['description'], 
                                             x['requirements'], x['benefits'], x['industry'], x['function']]), axis=1)
df['all_text'][0:10]
                          
from keras.preprocessing.text import one_hot

vocab_size = 1000
encoded_docs = [one_hot(d, vocab_size) for d in df['all_text'].values]
print(encoded_docs[0:1])
from keras.preprocessing.sequence import pad_sequences

max_length = 100
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs[0:1])
labels = df['fraudulent']
print(labels.value_counts())
print(len(labels))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.33, random_state=42, stratify=labels)
X_train.shape
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(vocab_size, 2, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
from keras.callbacks import EarlyStopping

model.fit(X_train, y_train, validation_split=0.3, epochs=50, verbose=1, callbacks=[EarlyStopping(monitor='val_loss')])
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))