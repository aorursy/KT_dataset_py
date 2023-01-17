# Load the libraries

import numpy as np

import pandas as pd



from pandas import DataFrame

from collections import Counter

from nltk.corpus import stopwords



from keras.preprocessing.text import Tokenizer

from keras.layers import Dense, Dropout,BatchNormalization,Input,Activation

from keras.models import Model

from keras import optimizers



from matplotlib import pyplot

# turn a text into clean tokens

def clean_text(text):

    # remove punctuation from the text

    text = text.replace("' ", " ' ")

    signs = set(',.:;"?!')

    prods = set(text) & signs

    if not prods:

        return text

    for sign in prods:

        text = text.replace(sign, ' {} '.format(sign) )

    # split into tokens by white space

    tokens = text.split()

    # remove remaining tokens that are not alphabetic

    tokens = [word for word in tokens if word.isalpha()]

    # filter out stop words

    stop_words = set(stopwords.words('english'))

    tokens = [w for w in tokens if not w in stop_words]

    # filter out short tokens

    tokens = [word for word in tokens if len(word) > 1]

    return tokens

#Define column names

TEXT ="text"

AUTHOR = "author"



# Create fucntion to load words into vocab



def add_text_to_vocab(X_train, vocab):

    for text in X_Train[TEXT]:

      tokens = clean_text(text)

      # update counts

      vocab.update(tokens)

 
# Read text from training data

X_Train= pd.read_csv("../input/train.csv")



# define vocab

vocab = Counter()



# add all text to vocab

add_text_to_vocab(X_Train,vocab)

# keep tokens with a min occurrence 

min_occurane = 2

tokens = [k for k,c in vocab.items() if c >= min_occurane]
# save list to file

def save_list(lines, filename):

    # convert lines to a single blob of text

    data = '\n'.join(lines)

    # open file

    file = open(filename, 'w')

    # write text

    file.write(data)

    # close file

    file.close()



# save tokens to a vocabulary file

save_list(tokens, 'vocab.txt')
# load file into memory

def load_file(filename):

    # open the file as read only

    file = open(filename, 'r')

    # read all text

    text = file.read()

    # close the file

    file.close()

    return text



# load the vocabulary back into memory to be used for model training

vocab_filename = 'vocab.txt'

vocab = load_file(vocab_filename)

vocab = vocab.split()

vocab = set(vocab)
# create the tokenizer

tokenizer = Tokenizer()



# prepare bag of words encoding of docs

def prepare_data(train_docs, mode):

    # fit the tokenizer on the documents

    tokenizer.fit_on_texts(train_docs)

    # encode training data set

    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)

    return Xtrain

# Create a neural network model

def create_model(xtrain):

    

    X_input = Input(shape=(xtrain.shape[1],))

   

    # dropout on input layer

    X = Dropout(0.5)(X_input)

    

    # Dense -> BN -> RELU-> Dropout Block applied to X - First set

    X = Dense(900, kernel_initializer='he_normal', name='D0')(X)

    X = BatchNormalization(axis=1, name='bn0')(X)

    X = Activation('relu')(X)

    X = Dropout(0.5)(X)



    # Dense -> BN -> RELU-> Dropout Block applied to X - Second set

    X = Dense(600, kernel_initializer='he_normal', name='D1')(X)

    X = BatchNormalization(axis=1, name='bn1')(X)

    X = Activation('relu')(X)

    X = Dropout(0.5)(X)

    

    # Dense -> BN -> RELU-> Dropout Block applied to X - Third set

    X = Dense(300, kernel_initializer='he_normal', name='D2')(X)

    X = BatchNormalization(axis=1, name='bn2')(X)

    X = Activation('relu')(X)

    X = Dropout(0.5)(X)



    # output layer with softmax function for 3 classes prediction

    X = Dense(3, kernel_initializer='he_normal', activation='softmax')(X)



    Spookymodel = Model(inputs=X_input, outputs=X, name='SpookyAuthor')

    

    return Spookymodel
#Prepare data and and create model 

# You can try with other modes 'binary','count','tfidf'

mode = 'freq'



#Training 

train_texts = X_Train[TEXT]

# Training labels (coverted into seperate columsn for each other with 0,1)

ytrain = np.array(pd.get_dummies(X_Train[AUTHOR]))



# prepare data for mode

xtrain = prepare_data(train_texts, mode)



# model defination creation

Spookymodel = create_model(xtrain)

    
# Summarize the model

Spookymodel.summary()
optimzer = optimizers.Adamax(lr=0.05, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.001)



# compile network

Spookymodel.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])



# fitnetwork

train_history = Spookymodel.fit(x= xtrain, y=ytrain, epochs=5,verbose=2,batch_size=64, validation_split=0.2)



# Plot the training and validation loss at each epoch

loss = train_history.history['loss']

val_loss = train_history.history['val_loss']

pyplot.plot(loss)

pyplot.plot(val_loss)

pyplot.legend(['loss', 'val_loss'])

pyplot.show()
# Function to predict the author class on test data

def predict_author(text, vocab,tokenizer, model):

    # clean

    tokens = clean_text(text)

    # filter by vocab

    tokens = [w for w in tokens if w in vocab]

    # convert to line

    line = ' '.join(tokens)

    # encode

    encoded = tokenizer.texts_to_matrix([line], mode='freq')

    # prediction

    yhat = model.predict(encoded, verbose=0)

    return yhat

# Load test data into dataframe

X_sub = pd.read_csv("../input/test.csv")



# Initilize prediction matrix 

y_pred = np.zeros((X_sub.shape[0],3))



# Predict for each  sample in test dataset

i = 0

for text in X_sub[TEXT]:

    y_pred[i]=predict_author(text,vocab,tokenizer,Spookymodel)

    i +=1

#Creating submission datafram

submission = pd.DataFrame(y_pred,dtype=float)

submission=submission.rename(index=int, columns={0: "EAP", 1: "HPL", 2: "MWS"})

submission.insert(0,'id',X_sub["id"])





#Save as CSV file for submission

submission.to_csv("sub.csv",sep=',', encoding='utf-8')