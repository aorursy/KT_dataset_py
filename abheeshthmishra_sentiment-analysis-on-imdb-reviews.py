from keras.datasets import imdb
((XT,YT),(Xt,Yt)) = imdb.load_data(num_words=10000)   #XT training # Xt testing
len(Xt),len(XT)
print(XT[0])
word_idx = imdb.get_word_index()
# print(word_idx.items())    # you run this cell to see the output
idx_word = dict([value,key] for (key,value) in word_idx.items())
# print(idx_word.items())    # you can also run this cell to see the output
actual_review = ' '.join([idx_word.get(idx-3,'#') for idx in XT[0]])
print(actual_review)
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt
##next step ----> Vectorize the data

## Vocab size --> 10,000 we will make sure every sentence is represented by a vector of len 10,000 [0000010001001011...]





def  vectorize_sentences(sentences,dim = 10000):

  outputs = np.zeros((len(sentences),dim))





  for i,idx in enumerate(sentences):

    outputs[i,idx] = 1



  return outputs
X_train  = vectorize_sentences(XT)

X_test = vectorize_sentences(Xt)
print(X_train.shape)

print(X_test.shape)
print(X_train[0])
Y_train  = np.asarray(YT).astype('float32')

Y_test = np.asarray(Yt).astype('float32')
from keras import models

from keras.layers import Dense
# define the model

model  = models.Sequential()

model.add(Dense(16,activation = 'relu' , input_shape = (10000,)))

model.add(Dense(16,activation = 'relu'))

model.add(Dense(1,activation = 'sigmoid'))
# here we are compiling

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy']) # you can use adam insted of rmsprop
model.summary()
x_val = X_train[:5000]

x_train_new = X_train[5000:]



y_val = Y_train[:5000]

y_train_new = Y_train[5000:]
hist = model.fit(x_train_new,y_train_new,epochs = 4,batch_size=512,validation_data =(x_val,y_val))
h = hist.history
plt.plot(h['val_loss'],label = 'validation loss')

plt.plot(h['loss'],label = 'training loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()

plt.style.use('seaborn')
plt.plot(h['val_accuracy'],label = 'validation Acc')

plt.plot(h['accuracy'],label = 'training Acc')

plt.xlabel('epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

plt.style.use('seaborn')
h = hist.history
# let's calculate accuracy

model.evaluate(X_test,Y_test)[1]
model.evaluate(X_train,Y_train)[1]