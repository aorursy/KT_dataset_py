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
# from keras.models import Model



# from keras.layers import Input, Dense, Flatten

# from keras.layers import Conv1D, MaxPooling1D

# from keras.layers import Embedding



from keras.preprocessing.text import Tokenizer

# from keras.preprocessing.text import text_to_word_sequence

# from keras.preprocessing.sequence import pad_sequences



# from keras.utils import to_categorical



# from pandas import DataFrame, concat, read_csv



from keras.models import Sequential

from keras.layers import Dense, LSTM 



from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

# from sklearn.metrics import mean_squared_error

# import nltk

# from nltk.corpus import stopwords

# from sklearn.metrics import confusion_matrix

# import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/train.csv')

df_train.head(5)
df_test = pd.read_csv('../input/test.csv')

df_test.head()
df_train.tail()
df_train.isnull().sum()
df_train[df_train.isnull().any(axis=1)]
# coverting NAN to str nan, and to make row as type str

df_train.converse=df_train.converse.astype(str)

df_train.isnull().sum()
# Labels count

num_labels = len(set(df_train['categories']))

num_labels
train_size = int(len(df_train) * .8)

train_converse = df_train['converse'][:train_size]

train_categories = df_train['categories'][:train_size]

val_converse = df_train['converse'][train_size:]

val_categories = df_train['categories'][train_size:]
print(train_size)

train_converse.head()
vocab_size = 1000

t = Tokenizer(num_words=vocab_size)

t.fit_on_texts(train_converse)

print(t.document_count)
# matrix

x_train = t.texts_to_matrix(train_converse)
#validation data

x_val = t.texts_to_matrix(val_converse)
##dummyfy the labels

encoder = LabelBinarizer()

encoder.fit(train_categories)

y_train = encoder.transform(train_categories)

y_val = encoder.transform(val_categories)

print(len(y_train))

print(len(y_val))
#Neural Network

model = Sequential()

model.add(Dense(350, input_shape=(vocab_size,) , activation='relu'))

model.add(Dense(num_labels,activation='softmax'))
batch_size = 10

model.compile(loss='categorical_crossentropy', 

              optimizer='adam',   ## 'sgd', 'rmsprop', adam

              metrics=['accuracy'])

history = model.fit(x_train, y_train, 

                    batch_size=batch_size, 

                    epochs=2, 

                    verbose=1, 

                    validation_split=0.1)
# summarize the model

print(model.summary())
score = model.evaluate(x_val, y_val, 

                       batch_size=batch_size, verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])
for i in range(5):    

    prediction = model.predict(np.array([x_val[i]]))

    text_labels = encoder.classes_ 

    predicted_label = text_labels[np.argmax(prediction[0])]

    print(val_converse.iloc[i][:50], "...")    

    print('Actual label:' + val_categories.iloc[i])

    print("Predicted label: " + predicted_label)

    print("==============================")
#test data

print(df_test.isnull().sum())

#print NA values

df_test[df_test.isnull().any(axis=1)]
df_test.converse=df_test.converse.astype(str)

x_test = t.texts_to_matrix(df_test['converse'])
Y_pred = model.predict(x_test)

y_pred =[]

for i in Y_pred:

    text_labels = encoder.classes_     

    y_pred.append(text_labels[np.argmax(i)])
id_list = list(range(1,len(y_pred)+1))

print(len(id_list))

labels=['id']

sample_out = pd.DataFrame( id_list, columns=labels)

sample_out['categories']=y_pred

sample_out.head()
sample_out.tail()