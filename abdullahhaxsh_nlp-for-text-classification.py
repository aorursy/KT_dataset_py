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

# ############ Loading and splitting data set ################# #



train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')



tweet = np.array(train_data['text'] , dtype = 'str')

target = np.array(train_data['target'])



x_train = tweet[0:6850]

y_train  = target[0:6850]



x_test = tweet[6851:]

y_test  = target[6851:]





print(x_train.dtype)

print(y_train.dtype)



print(x_train.shape , y_train.shape)

print(x_test.shape , y_test.shape)
# ############ Tokenizing / Sequencing / Padding ############### #



tokenizer = Tokenizer(num_words = 10000 , oov_token= "<OOV>")

tokenizer.fit_on_texts(x_train) ######### This only generates words data base of train #########

word_index = tokenizer.word_index



# ######### We dont tokenize test so that we dont have dat base of test words ############# #



sequence_train = tokenizer.texts_to_sequences(x_train)

sequence_test = tokenizer.texts_to_sequences(x_test)



pad_train = pad_sequences(sequence_train , padding = 'pre')

pad_test = pad_sequences(sequence_test , padding = 'pre')



# print(pad_train[0].size)

# print(pad_test[0].size)

# print(pad_test[0])



# print(x_train[0])

# print(sequence_train[0])

# print()

# print(x_test[0])

# print(sequence_test[0])
# ############# Model Creation ############### #



model = keras.Sequential([

keras.layers.Embedding( 10000 , 16 ),  # ######### Creates vectors in diff dimensions ############ #

keras.layers.GlobalAveragePooling1D(), ###### Sum up vectors to understand context ########



# #### Output and Dense layers #### #



keras.layers.Dense(24, activation='relu'),

keras.layers.Dense(1, activation='sigmoid'),



])



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()



model.fit(pad_train , y_train , epochs=50 , validation_data=(pad_test , y_test))



loss,  acc = model.evaluate(pad_test , y_test)



print(loss , acc)
# ######### Checking on validation data set ############ #



predictions = model.predict(pad_test)



for i in range(len(predictions)):

  print(  np.round(predictions[i] , 0)   , '\t' , y_test[i])
# ########### Making sample submission file ############### #



data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



sen = np.array(data['text'])

id_num = np.array(data['id'])





sequence_sen = tokenizer.texts_to_sequences(sen)

pad_sen = pad_sequences(sequence_sen , padding='pre')





# print(sen[0])

# print(sequence_sen[0])

# print(pad_sen[0])



# ############## Predictions ############## #



prediction = model.predict(pad_sen)

prediction = np.round(prediction , 0)

final_add = [int(j) for i in prediction for j in i]



print(final_add[0])

print(type(final_add[0]))

z  = { 

'id' :  id_num ,    

'target' : final_add , 

}







sample = pd.DataFrame(z)

sample.to_csv('Sample.csv')