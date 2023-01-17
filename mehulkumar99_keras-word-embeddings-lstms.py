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
"""loading the pre-processed training and testing dataset """



train_df  = pd.read_csv('/kaggle/input/preprocessed/train_cleaned.csv')

test_df = pd.read_csv('/kaggle/input/preprocessed/test_cleaned.csv')
train_df.head()
# Converting the raw text  to numeric tensors using pretrained glove 100d word embeddings (word level tokenization)



import keras

import numpy as np



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences





tokenizer = Tokenizer()                              

tokenizer.fit_on_texts(pd.concat([train_df['text_cleaned'],test_df['text_cleaned']]))   # builds the word index 

word_index = tokenizer.word_index

print('The size of the vocabulary in the training and testing set is : %d'%(len(word_index)))

vocab_size = max_features = len(word_index) 



# word_index.items()
# getting the 100d glove representation

embedding_dict_100d={}

with open('/kaggle/input/6b-glove/glove.6B.100d.txt','r', encoding='utf8') as f:

    for line in f:

        values=line.split(' ')

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict_100d[word]=vectors

f.close()



print('The size of the embedding index is : %d'%(len(embedding_dict_100d)))
# creating a glove word embedding matrix and setting the Embedding layer weights to the pretrained embedding matrix



embedding_dim  = 100



embedding_matrix = np.zeros((max_features+1, embedding_dim))



for word, i in word_index.items():

  if i < max_features:

    

    try:

      embedding_matrix[i] =embedding_dict_100d[word]

    except KeyError:  

      embedding_matrix[i] = np.zeros((1, embedding_dim))

embedding_matrix.shape
# creating the model architecture



from keras.models import Sequential

from keras.layers import Embedding , SimpleRNN, Dense, LSTM, SpatialDropout1D

from keras.initializers import Constant

maxlen = 25



model = Sequential()

model.add(Embedding(max_features+1,embedding_dim, embeddings_initializer=Constant(embedding_matrix),

                    input_length=maxlen,

                    trainable=False))



model.add(SpatialDropout1D(0.4))

model.add(LSTM(50, return_sequences=True))

model.add(SpatialDropout1D(0.4))

model.add(LSTM(50))

model.add(Dense(1, activation='sigmoid'))

model.summary()
# model is ready. next step is to prepare the data for feeding into the network.

# setting batch_size, maximum length allowed for a sequence 

from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split



batch = 64



training_samples = 7000

validation_samples = 613



sequences = tokenizer.texts_to_sequences(train_df['text_cleaned'])  # transform each tweets in to a sequence of integers

data = pad_sequences(sequences, maxlen=maxlen, truncating='post', padding='post')  # (longer sequence than threshold will be truncated and shorter sequence will be padded with zeros)

labels = np.asarray(train_df['target'])



print('Shape of label tensor: %d' %(labels.shape))

print('Shape of data tensor:', data.shape)



x_train,x_val,y_train,y_val = train_test_split(data, labels, test_size = 0.050,random_state = 0, shuffle = True ) 
print('Shape of x_train tensor:', x_train.shape)

print('Shape of x_val tensor:', x_val.shape)

print('Shape of y_train tensor:', y_train.shape)

print('Shape of y_val tensor:', y_val.shape)
#  training and evaluation 

from keras.optimizers import Adam



model.compile(optimizer = Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc'] )



history = model.fit(x_train, y_train,epochs=50  , batch_size=64, validation_data=(x_val, y_val),verbose=2)

model.save_weights('pre_trained_glove_model.h5')
test_sequences = tokenizer.texts_to_sequences(test_df['text_cleaned'])  # transform each tweets in to a sequence of integers

test_data = pad_sequences(test_sequences, maxlen=maxlen, truncating='post', padding='post')  # (longer sequence than threshold will be truncated and shorter sequence will be padded with zeros)



predictions = model.predict(test_data)

predictions = np.round(predictions).astype(int).reshape(3263)



# Creating submission file 

submission = pd.DataFrame({'id' : test_df['id'], 'target' : predictions})

submission.to_csv('final_submission.csv', index=False)



submission.head()
