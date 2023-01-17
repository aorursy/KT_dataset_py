# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!kaggle datasets download -d thanakomsn/glove6b300dtxt

#to use gglove6b300d for embedding
!git clone https://github.com/ipmob/NITD-machine-Learning-challange
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import re

import keras



import nltk 

nltk.download('punkt')



nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))
data = pd.read_csv('/content/NITD-machine-Learning-challange/data/train_file.csv')

test = pd.read_csv('/content/NITD-machine-Learning-challange/data/test_file.csv')

sample = pd.read_csv('/content/NITD-machine-Learning-challange/data/results_file.csv')



data.loc[data.Subjects.isnull(), 'Subjects'] = ''

test.loc[test.Subjects.isnull(), 'Subjects'] = ''



data['PublicationYear'] = data['PublicationYear'].astype('str').map(lambda x: " ".join(re.findall('\d{4}', x)))

test['PublicationYear'] = test['PublicationYear'].astype('str').map(lambda x: " ".join(re.findall('\d{4}', x)))

test = test.fillna({'Subjects': "", "Title": "", "Publisher": ""})

data = data.fillna({"PublicationYear":"","Publisher":""})



data['combined'] = data['Subjects'] + " " + data['Title'] + " " + data['PublicationYear'].astype('str') + " " + data['Publisher'] + " " + data['Checkouts'].astype('str')

test['combined'] = test['Subjects'] + " " + test['Title'] + " " + test['PublicationYear'].astype('str') + " " + test['Publisher'] + " " + data['Checkouts'].astype('str')



# data['combined'] = data.Subjects.str.cat(' ' + data.Title)

# test['combined'] = test_data['Subjects']



# data['combined'] = td2['sub']



dataMatType = data.MaterialType

data.drop(labels = ['MaterialType'], axis = 1, inplace = True)



data.combined = data.combined.apply(preprocess_text)

test.combined = test.combined.apply(preprocess_text)

'''

data.Subjects = data.Subjects.apply(preprocess_text)

data.Title = data.Title.apply(preprocess_text)

test.Subjects = test.Subjects.apply(preprocess_text)

test.Title = test.Title.apply(preprocess_text)

'''

flag = -1

labelDict = {}



for key in dataMatType.value_counts().keys():

  flag += 1

  labelDict[key] = flag

  

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import numpy as np



maxlen = 25                            



max_words = 1000                                    



tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(data.combined)



dataSequences = tokenizer.texts_to_sequences(data.combined)

testSequences = tokenizer.texts_to_sequences(test.combined)



data_word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(data_word_index))
from keras import preprocessing

x_train = preprocessing.sequence.pad_sequences(dataSequences, maxlen = 25)

x_test = preprocessing.sequence.pad_sequences(testSequences, maxlen = 25)



train_label = dataMatType.map(labelDict)



from keras.utils import to_categorical

labelDataBinary = to_categorical(train_label)



from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(dataMatType), dataMatType)
import os

glove_dir = '/content'



embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))



embedding_dim = 300



embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in data_word_index.items():

    if i < max_words:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector
reverseLabelDict = {}



for key, value in labelDict.items():

  reverseLabelDict[value] = key

def predict(model, x_test):

  predictions = model.predict(x_test)

  predictions = pd.Series([np.argmax(i) for i in predictions], index = test.index)

  predictions = predictions.map(reverseLabelDict)

  return predictions
from keras.layers import LSTM

from keras.models import Sequential

from keras.layers import Dense, Flatten, Embedding, SimpleRNN, Conv1D, MaxPooling1D

from keras.layers import Dropout

from keras.layers import Dense



model1 = Sequential()

model1.add(Embedding(1200, 1200, input_length=25))



'''model.add(Conv1D(256, 5, activation='relu'))

model.add(MaxPooling1D(pool_size=4))

'''





model1.add(Conv1D(128, 3,  activation='elu'))

model1.add(Dropout(0.6))



model1.add(LSTM(400, recurrent_dropout = 0.6, return_sequences = True))

model1.add(Dropout(0.4))



model1.add(LSTM(400, recurrent_dropout = 0.5))

model1.add(Dropout(0.2))



model1.add(Dense(8, activation='softmax'))



model1.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['acc'])

history = model1.fit(x_train, labelDataBinary,

                    epochs=16,

                    validation_split=0,

                    class_weight = class_weights)



model1.save('/content/m1.h5')

#0.86623 

#0.87633



predic1 = pd.Series(predict(model1, x_test))

submission = pd.concat([test.ID.astype(np.int), predic1], axis = 1)

submission.columns = ['ID' , 'MaterialType']

submission.to_csv('/content/submition_model1.1_add.csv', index = False, header = True)  

from keras.layers import CuDNNLSTM,CuDNNGRU

from keras.models import Sequential

from keras.layers import Dense, Flatten, Embedding, SimpleRNN, Conv1D, MaxPooling1D

from keras.layers import Dropout

from keras.layers import Dense

from keras import optimizers,callbacks

import keras

model6 = Sequential()

model6.add(Embedding(1600, 1600 ,input_length=25))





model6.add(CuDNNGRU(600, return_sequences = True))

#model.add(Dropout(0.6))

model6.add(Dropout(0.7))



model6.add(CuDNNGRU(500))

model6.add(Dropout(0.6))



model6.add(Dense(100, activation='elu'))

model6.add(Dense(50, activation='elu'))

model6.add(Dense(8, activation='softmax'))



adam = optimizers.adam(lr = 0.0014)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

model6.compile(optimizer=adam,

              loss='categorical_crossentropy',

              metrics=['acc'])

history = model6.fit(x_train, labelDataBinary,

                    epochs=16,

                    validation_split=0,

                    class_weight = class_weights,

                   callbacks=[reduce_lr])



model6.save('/content/model_morefeatures.h5')

#v1_0.87181 

#0.87181 

#0.87695

predic6 = pd.Series(predict(model6, x_test))

submission = pd.concat([test.ID.astype(np.int), predic6], axis = 1)

submission.columns = ['ID' , 'MaterialType']

submission.to_csv('/content/submition_model_add_v1.csv', index = False, header = True)
num_model = 6

predic1 = pd.Series(predict(model1, x_test))

predic2 = pd.Series(predict(model2, x_test))

predic3 = pd.Series(predict(model3, x_test))

predic4 = pd.Series(predict(model4, x_test))

predic5 = pd.Series(predict(model5, x_test))

predic6 = pd.Series(predict(model6, x_test))