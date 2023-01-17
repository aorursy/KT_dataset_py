import pandas as pd

import os
print(os.listdir('../input/nlpglovedataset'))

PATH = '../input/nlpglovedataset/'



# Data is review of amazon , yelp and imdb reviews

# glove.6B.*.txt are pre-trained weights
# Create one datafarme havinf data of all source

# New column created to distiguish data source



reviewDataPath = {'yelp': PATH +'yelp_labelled.txt',

                 'amazon': PATH +'amazon_cells_labelled.txt',

                 'imdb': PATH +'imdb_labelled.txt'}

reviewList = []



for source, filepath in reviewDataPath.items():

    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')

    # Add another column filled with the source name

    df['source'] = source 

    reviewList.append(df)



df = pd.concat(reviewList)
review_imdb = df[df['source'] == 'amazon']

review_imdb.info()
# Just a look at data

print(df.iloc[:10])
from sklearn.model_selection import train_test_split
# Creating model of YELP data onl



review_yelp = df[df['source'] == 'yelp']



sentences = review_yelp['sentence'].values



y = review_yelp['label'].values



sentences_train, sentences_test, y_train, y_test = train_test_split(

    sentences, y, test_size=0.25, random_state=1000)
#sentences_train.size
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=1000)



tokenizer.fit_on_texts(sentences_train)



X_train = tokenizer.texts_to_sequences(sentences_train)

X_test = tokenizer.texts_to_sequences(sentences_test)

# Adding 1 because of reserved 0 index

# The indexing is ordered after the most common words in the text, 

# which you can see by the word the having the index 1. 

# It is important to note that the index 0 is reserved 

# and is not assigned to any word. This zero index is used for padding,

# because every statement is not of same size



vocab_size = len(tokenizer.word_index) + 1 



print(vocab_size)
# Increasing vocab size by 1 as need to make room for '0' index

vocab_size = len(tokenizer.word_index) + 1
# Lets look at top 5 sentence 

print(sentences_train[:6])
# Lets look at top 5 sentence toeknized 

print(X_train[0])

print(X_train[1])

print(X_train[2])

print(X_train[3])

print(X_train[4])

print(X_train[5])
from keras.preprocessing.sequence import pad_sequences
# maxlen parameter to specify how long the sequences should be. 

#This cuts sequences that exceed that number.



maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train[1])
print(X_train[4])
from keras.models import Sequential

from keras import layers
model = Sequential()
# vocab size is 1750 

# input_length is size of review text after tokenization and pad sequance

embedding_dim = 50





model.add(layers.Embedding(input_dim=vocab_size,

                           output_dim=embedding_dim,

                           input_length=maxlen))



model.add(layers.Flatten())

model.add(layers.Dense(6, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))





model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

history = model.fit(X_train, y_train,

                    epochs=25,verbose=True,

                    validation_data=(X_test, y_test),

                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy: {:.4f}".format(accuracy))

import numpy as np

phrase = "good food ,will come again"

#phrase = "bad service"

#phrase = "asdasd"



# ------------ALERT-------------------------

#  Need to use same tokenizer object 

#  which was used to tokenize training data

# ------------------------------------------

tokens = tokenizer.texts_to_sequences([phrase])

pad_tokens = pad_sequences(tokens, padding='post', maxlen=maxlen)



print(tokens)

print(pad_tokens)
val = model.predict_classes(pad_tokens)   

print(val)
def predictSentiments ( indexvalue):

    

    reviewSentiment = ''

    

    if (val[0][0] == 0):

        reviewSentiment = 'Customer is gone forever,'

    else:

        reviewSentiment = 'you got back your customer'



    return reviewSentiment;
print(predictSentiments(val[0][0]))
from keras.models import load_model

import pickle



# Creates a HDF5 file 'my_model.h5'

model.save('my_model.h5')



# Deletes the existing model

#del model  





# saving tokenizer 

with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



# loading

with open('tokenizer.pickle', 'rb') as handle:

    tokenizer_saved = pickle.load(handle)



# Returns a compiled model identical to the previous one

model_saved = load_model('my_model.h5')
#review_sen = "good food ,will come again"

review_sen = "bad service"



tokens_sen = tokenizer_saved.texts_to_sequences([review_sen])

pad_tokens_sen = pad_sequences(tokens_sen, padding='post', maxlen=maxlen)



print(tokens_sen)

print(pad_tokens_sen)
val = model_saved.predict_classes(pad_tokens_sen)

print(predictSentiments(val[0][0]))
model2 = Sequential()



model2.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))



# remove flatten and use global max pool

model2.add(layers.GlobalMaxPool1D())



model2.add(layers.Dense(6, activation='relu'))

model2.add(layers.Dense(1, activation='sigmoid'))

model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model2.summary()
history2 = model2.fit(X_train, y_train,

                    epochs=25,verbose=True,

                    validation_data=(X_test, y_test),

                    batch_size=10)
loss, accuracy = model2.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))



loss, accuracy = model2.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy: {:.4f}".format(accuracy))



# Creating embedding from pre-trained GloVe 

# GloVe is trained on millions of senetece

# We need to take weights of wrods which 

# exist in our training data only

# This method takes words from training data and

# copy weight of word from GloVe to a new array

# which we will use as word embeding



def create_embedding_matrix(filepath, word_index, embedding_dim):

    

    vocab_size = len(word_index) + 1 

    # Adding again 1 because of reserved 0 index

    

    embedding_matrix = np.zeros((vocab_size, embedding_dim))



    with open(filepath) as file:

        for line in file:

            word, *vector = line.split()

            if word in word_index:

                idx = word_index[word]

                #print("{} {} ".format(word,idx))

                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
# Lets take 50-dimesional weights



embedding_dim = 50



filePath = PATH + 'glove.6B.50d.txt'



embedding_matrix = create_embedding_matrix(filePath,

                                           tokenizer.word_index, 

                                           embedding_dim)
print(embedding_matrix[0:2])
model3 = Sequential()



model3.add(layers.Embedding(vocab_size, 

                            embedding_dim,

                            weights=[embedding_matrix], # Change is here

                            input_length=maxlen,

                            trainable=True)) # Make it False to check model perfromance

#model3.add(layers.Conv1D(128, 5, activation='relu'))

model3.add(layers.GlobalMaxPool1D())



model3.add(layers.Dense(5, activation='relu'))

model3.add(layers.Dense(1, activation='sigmoid'))

model3.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model3.summary()
history3 = model3.fit(X_train, y_train,

                    epochs=50,verbose=True,

                    validation_data=(X_test, y_test),

                    batch_size=10)
loss, accuracy = model3.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model3.evaluate(X_test, y_test, verbose=False)

print("Test Accuracy: {:.4f}".format(accuracy))
