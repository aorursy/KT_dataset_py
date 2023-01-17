import numpy as np

import pandas as pd

import tensorflow

import keras

import sklearn.model_selection

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/new_dat.csv')

print( data.head() )

print( data.isnull().sum() )
text = np.array( data['Text'] )

target = np.array( data['Target'] )

print(text.shape , target.shape)



# ################# Spliting - DATA ############### #



x_train , y_test , x_lab , y_lab = sklearn.model_selection.train_test_split(text , target , test_size = 0.1)



# x_train , y_test =  text[ : 20000] , text[20000 : ]

# x_lab , y_lab = target[ : 20000] , target[ 20000 : ]



print(x_train.shape , y_test.shape)

print(x_lab.shape , y_lab.shape)
tokenizer = Tokenizer( num_words=10000 , oov_token='<OOV>' )

tokenizer.fit_on_texts(x_train)

word_index = tokenizer.word_index



# for i in word_index.items():

#   print(i)



# ###### We make sequence / padding of testing and training data ######### #



sequence_train = tokenizer.texts_to_sequences(x_train)

sequence_test = tokenizer.texts_to_sequences(y_test)



pad_train = pad_sequences(sequence_train , padding = 'pre') 

pad_test = pad_sequences(sequence_test , padding = 'pre')
model = keras.Sequential()

model.add( keras.layers.Embedding( 10000 , 16) ) # Creates vectors in 16 dimensions

model.add( keras.layers.GlobalAveragePooling1D() ) # fIND resultant vector

model.add( keras.layers.Dense( 24 , activation='relu') )

model.add( keras.layers.Dense( 1 , activation='sigmoid') )





model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
model.fit( pad_train , x_lab , epochs=10 , validation_data=(pad_test , y_lab) )

loss,  acc = model.evaluate(pad_test , y_lab)



print(loss , acc)
model.save("MODEL.h5")
predictions = model.predict( pad_test )

predictions = np.round(predictions , 0)

for i in range(50):

  print( predictions[i][0] , y_lab[i] )