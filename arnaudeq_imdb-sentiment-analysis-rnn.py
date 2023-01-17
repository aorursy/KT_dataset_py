%matplotlib inline

import keras

import numpy as np

from keras import backend as K

from keras.utils.data_utils import get_file

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.models import Sequential, Model

from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional

from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.regularizers import l2, l1

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import *

from keras.optimizers import SGD, RMSprop, Adam

from keras.metrics import categorical_crossentropy, categorical_accuracy

from keras.layers.convolutional import *

from keras.preprocessing import image, sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3

from keras import applications
import numpy as np

from keras.datasets import imdb as keras_imdb
#imdb = np.load(open("../input/imdb.npz", "rb"), allow_pickle=True)

#x_train = imdb["x_train"]

#y_train = imdb["y_train"]

#x_test = imdb["x_test"]

#y_test = imdb["y_test"]
import numpy as np

# save np.load

np_load_old = np.load



# modify the default parameters of np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



# call load_data with allow_pickle implicitly set to true

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=None,

                                                       skip_top=0,

                                                       maxlen=None,

                                                       seed=113,

                                                       start_char=1,

                                                       oov_char=2,

                                                       index_from=3)



# restore np.load for future normal usage

np.load = np_load_old
import nltk
sentence = "            Hello,  my name's Chien #test           "
l = sentence.split()
" ".join(l)
tokenizer = nltk.tokenize.TweetTokenizer() 
tokenizer.tokenize(sentence)
print(x_train[1][:10])
print(x_train.shape)
print(len(x_train[0]))
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id["horrible"]
list(word_to_id.keys())[:10]
# Add terms to our vocabulary

word_to_id = {word : (word_id + 3) for word, word_id in keras.datasets.imdb.get_word_index().items()}  # Add an offset of 3 to leave room for the new terms

word_to_id["<PAD>"] = 0

word_to_id["<START>"] = 1

word_to_id["<UNK>"] = 2



# Also define the opposite mapping

id_to_word = {word_id: word for word, word_id in word_to_id.items()}
def get_review(x, i):

    return " ".join(id_to_word[id_] for id_ in x[i])



def get_label(y, i):

    if y[i] == 1:

        return "Positive"

    else:

        return "Negative"
for review_index in range(10):

    review = get_review(x_train, review_index)

    label = get_label(y_train, review_index)

    print("*" * 50)

    print(f"Review index: {review_index}")

    print(f"Label: {label}")

    print("-" * 50)

    print(review)

    print("*" * 50)
vocab_size = 5000

trn = [np.array([i if i<vocab_size-1 else vocab_size-1 for i in s]) for s in x_train]

test = [np.array([i if i<vocab_size-1 else vocab_size-1 for i in s]) for s in x_test]
lens = np.array(list(map(len, trn)))

print('Maximum text length:', lens.max(),' -- Minimum length:', lens.min(), '-- Mean length of text:',lens.mean())
# we'll pad all inputs to obtain homogeneous inputs of dim 500

seq_len = 500



trn = sequence.pad_sequences(trn, maxlen = seq_len,value=0, padding="post", truncating="post")

test = sequence.pad_sequences(test, maxlen = seq_len,value=0, padding="post", truncating="post")
trn.shape
get_review(trn, 1)
def MLP():

    model = Sequential()

    model.add(Embedding(vocab_size,32,input_length=seq_len))

    

    model.add(Flatten())

    model.add(Dense(100,activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(100,activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(1,activation='sigmoid'))



    model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    print(model.summary())

    

    return model



model = MLP()  
model.fit(trn, y_train, validation_data=(test, y_test), epochs=4, batch_size=512)
scores = model.evaluate(test,y_test,verbose=0)

print('loss: ', scores[0],'- accuracy: ', scores[1])
def CNN():

    model = Sequential()

    model.add(Embedding(vocab_size, 32, input_length=seq_len))

    

    #model.add(Dropout(0.2))

    model.add(Conv1D(64, 5, padding='same', activation='relu'))

    model.add(Dropout(0.2))

    model.add(MaxPooling1D())

    

    model.add(Flatten())

    

    model.add(Dense(100, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    print(model.summary())

    

    return model



model = CNN()
model.fit(trn, y_train, validation_data=(test, y_test), epochs=2)
scores = model.evaluate(test,y_test,verbose=0)

print('loss: ', scores[0],'- accuracy: ', scores[1])
from keras.layers import concatenate
graph_in = Input ((vocab_size, 32))

convs = [ ] 

for fsz in range (3, 6): 

    x = Conv1D(64, fsz, padding='same', activation="relu")(graph_in)

    x = MaxPooling1D()(x) 

    x = Flatten()(x) 

    convs.append(x)

out = concatenate(convs)

graph = Model(graph_in, out)

graph.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])
graph.summary()
def multisize_CNN():

    model = Sequential()

    model.add(Embedding(vocab_size, 32,input_length=seq_len))

    model.add(Dropout (0.2))

    model.add(graph)

    model.add(Dropout(0.5))

    model.add(Dense(100, activation="relu"))

    model.add(Dropout(0.7))

    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    print(model.summary())

    

    return(model)



model = multisize_CNN()
model.fit(trn, y_train, validation_data=(test, y_test), epochs=5, batch_size=512)
scores = model.evaluate(test,y_test,verbose=0, batch_size=512)

print('loss: ', scores[0],'- accuracy: ', scores[1])
predictions = model.predict(test[:10],1)

predictions = np.round(predictions).astype('int')
for review_index in range(10):

    review = get_review(x_train, review_index)

    prediction = get_label(predictions, review_index)

    truth = get_label(y_test, review_index)

    

    print("*" * 50)

    print(f"Review index: {review_index}")

    print(f"Prediction: {label}  --- Truth: {truth}")

    print("-" * 50)

    print(review)

    print("*" * 50)
text = get_text(x_test, 10)

preds = get_label_txt(predictions[:10])

true = get_label_txt(y_test[0:10])
for i in range(40):

    print('*******************************************************************************')

    print('TEXT n°', i + 1, ' -- TRUE label:', true[i], ' -- PREDICTED label:', preds[i])

    print('-------------------------------------------------------------------------------')

    print(text[i])

    print('*******************************************************************************')
def RNN_LSTM():

    model = Sequential()

    model.add(Embedding(vocab_size,5,input_length=seq_len))

    model.add(LSTM(50))

    model.add(Dropout(0.25))

    model.add(Dense(1,activation='sigmoid'))



    model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    print(model.summary())

    

    return model



model = RNN_LSTM()       
model.fit(trn, y_train, validation_data=(test, y_test), epochs=4, batch_size=512)
scores = model.evaluate(test,y_test,verbose=0, batch_size=512)

print('loss: ', scores[0],'- accuracy: ', scores[1])
predictions = model.predict(test[:10],1)

predictions = np.round(predictions).astype('int')
n_texts = 20
text = get_text(x_test, n_texts)

preds = get_label_txt(predictions[:n_texts])

true = get_label_txt(y_test[:n_texts])
for i in range(n_texts):

    print('*******************************************************************************')

    print('TEXT n°', i + 1, ' -- TRUE label:',  true[i], ' -- PREDICTED label:', preds[i])

    print('-------------------------------------------------------------------------------')

    print(text[i])

    print('*******************************************************************************')
def RNN_GRU():

    model = Sequential()

    model.add(Embedding(vocab_size,5,input_length=seq_len))

    model.add(GRU(50))

    model.add(Dropout(0.25))

    model.add(Dense(1,activation='sigmoid'))



    model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    print(model.summary())

    

    return model



model = RNN_GRU()  
model.fit(trn, y_train, validation_data=(test, y_test), epochs=4, batch_size=512)
scores = model.evaluate(test,y_test,verbose=0)

print('loss: ', scores[0],'- accuracy: ', scores[1])
predictions = model.predict(test[:10],1)

predictions = np.round(predictions).astype('int')
text = get_text(x_test, 10)

preds = get_label_txt(predictions[:10])

true = get_label_txt(y_test[:10])
for i in range(10):

    print('*******************************************************************************')

    print('TEXT n°', i + 1, ' -- TRUE label:', true[i], ' -- PREDICTED label:', preds[i])

    print('-------------------------------------------------------------------------------')

    print(text[i])

    print('*******************************************************************************')