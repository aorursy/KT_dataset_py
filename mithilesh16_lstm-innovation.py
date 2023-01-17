import numpy

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf



import io
datasetx = pd.read_csv('../input/receiver-design-data-3/data_coloumn_new_3.csv')

datasety = pd.read_csv('../input/receiver-design-data-3/ri_coloumn_new_3.csv')
numpy.random.seed(7)



X = datasety.iloc[:, 0:1].values

y = datasetx.iloc[:, 0:1].values





ri = datasety.iloc[:, 0:1].values

tx = datasetx.iloc[:, 0:1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size = 0.2, random_state = 0)

np.reshape(X_train, (-1,1))

np.reshape(y_train, (-1,1))



print(X_train.shape[0])
print(X_train)
!pip3 install tensorflow
import tensorflow as tf



depth = y_train.shape[0]

indices = y_train

one_hot_matrix = tf.one_hot(indices,depth)

#one_hot_matrix = tf.one_hot( 

#	y_train, C, on_value = 1.0, off_value = 0.0, axis =-1) 



#sess = tf.Session() 



#one_hot = sess.run(one_hot_matrix) 



#sess.close() 



# output is of dimension 5 x 5 

print(one_hot_matrix)



from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.embeddings import Embedding





# integer encode the documents

vocab_size = X_train.max()+1

print(np.shape(vocab_size))

#encoded_docs = [one_hot(d, vocab_size) for d in X_train.ravel()]

#print(encoded_docs)

# pad documents to a max length of 4 words

max_length = 1

#padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

#print(padded_docs)

# define the model

#model = Sequential()

#model.add(Embedding(vocab_size, 1, input_length=max_length))

#model.add(Flatten())

#model.add(Dense(1, activation='sigmoid'))

# compile the model

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model

#print(model.summary())

# fit the model

#model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model

#loss, accuracy = model.evaluate(padded_docs, y_train, verbose=0)

#print('Accuracy: %f' % (accuracy*100))
embed_dim = 128

lstm_out = 200

batch_size = 32





# create the model

#embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(9999, 1, input_length=1))

model.add(LSTM(500))



#model.add(Flatten())

#model.add(Dense(5, activation='sigmoid'))

#model.add(Dense(5, activation='sigmoid'))

#model.add(Dense(5, activation='sigmoid'))

#model.add(Dense(5, activation='sigmoid'))

#model.add(Dense(2, activation='softmax'))

model.add(Dense(5, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, epochs=15, batch_size=20)





# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)

predict = model.predict(X_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))



y_pred=model.predict(X_test)



print("before y pred: " , y_pred)



for i in range(len(y_pred)):

    if(y_pred[i] < 0.5):

      y_pred[i] = 0

    else:

      y_pred[i] = 1



no_errors = (y_pred != y_test)

no_errors = no_errors.astype(int).sum()

ber = no_errors / len(y_test)



print("total no of errors: ", no_errors)



print("BER: ", ber)
pd_FM = {}

idx=0

X_FM_lb=[]

for snr in range(-5, 50, 1):

  X_FM_lb.append(ri[idx:idx+100])

  idx=idx+100



#print(X_FM_lb)





for snr in range(-5, 50, 1):

  y_snr = np.ones((X_FM_lb[snr+5].shape[0], 1))

  scores = model.evaluate(X_FM_lb[snr+5], y_snr)

  print("At SNR = " + str(snr) + "\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

  pd_FM[snr] =(1- scores[1])/2

  

plt.plot(range(-5, 50, 1), list(pd_FM.values()))





plt.yscale('log')

plt.xlabel('SNR Range')

plt.ylabel('BER')

plt.grid()

plt.legend(loc='upper right',ncol = 1)

plt.show()