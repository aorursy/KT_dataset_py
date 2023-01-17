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
data=pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding = "ISO-8859-1")

data.head()
data.tail()
sms=data["v2"]

y=data["v1"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sms, y, test_size=0.33, random_state=2001)
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(x_train)

X_train=tokenizer.texts_to_sequences(x_train)

X_test=tokenizer.texts_to_sequences(x_test)
maxlen=-11

for sms in X_train:

    candidatemax=len(sms)

    if maxlen<=candidatemax:

        maxlen=candidatemax

print(maxlen)        
maxlentest=-11

for sms in X_test:

    candidatemax=len(sms)

    if maxlentest<=candidatemax:

        maxlentest=candidatemax

print(maxlentest)       
from keras.preprocessing.sequence import pad_sequences

maxlen=200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
vocab_size = len(tokenizer.word_index) + 1 

print(vocab_size)
from keras.models import Sequential

from keras import layers



embedding_dim = 100



model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, 

                           output_dim=embedding_dim, 

                           input_length=maxlen))

model.add(layers.Flatten())

model.add(layers.Dense(100, activation='relu'))

model.add(layers.Dense(50, activation='relu'))

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

Y_train=le.fit_transform(y_train)

Y_test=le.transform(y_test)
model.fit(X_train, Y_train,

                    epochs=15,

                    verbose=True,

                    validation_data=(X_test, Y_test),

                    batch_size=10)

loss, accuracy = model.evaluate(X_train, Y_train)
from sklearn.metrics import confusion_matrix

ypred=model.predict(X_test)

print(confusion_matrix(y_true=Y_test,y_pred=ypred>0.5))
le.classes_
le.inverse_transform([0])
tn, fn, fp, tp = confusion_matrix(y_true=Y_test,y_pred=ypred>0.5).ravel()
fpr=fp/(fp+tn)

print("false positive rate: ",fpr)