from keras.datasets import imdb



#filter out top 10000 used words

vocab_size = 10000 
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# save np.load

np_load_old = np.load



# modify the default parameters of np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



# call load_data with allow_pickle implicitly set to true

#load dataset as a list of ints

# vocab_size is no.of words to consider from the dataset, ordering based on frequency.

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)



# restore np.load for future normal usage

np.load = np_load_old
#Maximum sequence length

#number of words used from each review

maxlen = 300  



#make all sequences of the same length using pad_sequences

X_train = pad_sequences(X_train, maxlen=maxlen)

X_test =  pad_sequences(X_test, maxlen=maxlen)
print(X_train[8],y_train[8])



#Here the X_train is sequence representing the most commonly used words in the overall data say 1:1st commonly used word,171:171st commonly used word in the data.
print(X_train[558],y_train[558])
from keras.models import Model, Sequential

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D
def create_seq_model():

  model = Sequential()

  #Here the 10000 is some random number, which is much larger than needed to reduce the probability of collisions from the hash function

  #The number 10k should be greater than the total no of letters in each sequence

  model.add(Embedding(10000,256,input_length=300))

  model.add(Bidirectional(LSTM(32, return_sequences = True)))

  model.add(GlobalMaxPool1D())

  model.add(Dense(20, activation="relu"))

  model.add(Dropout(0.05))

  model.add(Dense(1, activation="sigmoid"))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model
seq_nlp_model=create_seq_model()



# summarize the model



print(seq_nlp_model.summary())
''' 

batch_size = 100

epochs = 3

'''





batch_size = 10

epochs = 3



# fit the model

seq_nlp_model.fit(X_test,y_test, batch_size=batch_size, epochs=epochs, validation_split=0.2)
from sklearn.metrics import f1_score, confusion_matrix
y_pred = seq_nlp_model.predict(X_test)





print('F1-score: {0}'.format(f1_score(y_pred, y_test)))

print('Confusion matrix:')

confusion_matrix(y_pred, y_test)