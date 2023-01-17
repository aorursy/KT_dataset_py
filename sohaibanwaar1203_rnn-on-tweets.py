import os
print(os.listdir("../input/glove-global-vectors-for-word-representation"))
# import all libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.convolutional import Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import spacy
nlp=spacy.load("en")

#load the dataset
train=pd.read_csv("../input/tweets-sentimental-analysis/tweets.csv" , encoding= "latin-1")
Y_train = train[train.columns[0]]
X_train = train[train.columns[5]]

# split the data into test and train
from sklearn.model_selection import train_test_split
trainset1x, trainset2x, trainset1y, trainset2y = train_test_split(X_train.values, Y_train.values, test_size=0.02,random_state=42 )
trainset2y=pd.get_dummies(trainset2y)

# function to remove stopwords
def stopwords(sentence):
    new=[]
    sentence=nlp(sentence)
    for w in sentence:
        if (w.is_stop == False) & (w.pos_ !="PUNCT"):
            new.append(w.string.strip())
        c=" ".join(str(x) for x in new)
    return c

# function to lemmatize the tweets
def lemmatize(sentence):
    sentence=nlp(sentence)
    str=""
    for w in sentence:
        str+=" "+w.lemma_
    return nlp(str)

#loading the glove model
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print ("Done."),len(model),(" words loaded!")
    return model
# save the glove model
model=loadGloveModel("../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt")

#vectorising the sentences
def sent_vectorizer(sent, model):
    sent_vec = np.zeros(200)
    numw = 0
    for w in sent.split():
        try:
            sent_vec = np.add(sent_vec, model[str(w)])
            numw+=1
        except:
            pass
    return sent_vec

#obtain a clean vector
cleanvector=[]
for i in range(trainset2x.shape[0]):
    document=trainset2x[i]
    document=document.lower()
    document=lemmatize(document)
    document=str(document)
    cleanvector.append(sent_vectorizer(document,model))

#Getting the input and output in proper shape
cleanvector=np.array(cleanvector)
cleanvector =cleanvector.reshape(len(cleanvector),200,1)

#tokenizing the sequences
tokenizer = Tokenizer(num_words=16000)
tokenizer.fit_on_texts(trainset2x)
sequences = tokenizer.texts_to_sequences(trainset2x)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=15, padding="post")
print(data.shape)

#reshape the data and preparing to train
data=data.reshape(len(cleanvector),15,1)
from sklearn.model_selection import train_test_split
trainx, validx, trainy, validy = train_test_split(data, trainset2y, test_size=0.3,random_state=42 )
#calculate the number of words
nb_words=len(tokenizer.word_index)+1

#obtain theembedding matrix
embedding_matrix = np.zeros((nb_words, 200))
for word, i in word_index.items():
    embedding_vector = model.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

trainy=np.array(trainy)
validy=np.array(validy)

#building a simple RNN model
def modelbuild():
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=(15,1)))
    keras.layers.embeddings.Embedding(nb_words, 15, weights=[embedding_matrix], input_length=15,
    trainable=False)
    model.add(keras.layers.recurrent.SimpleRNN(units = 100, activation='relu',
    use_bias=True))
    model.add(keras.layers.Dense(units=1000, input_dim = 2000, activation='sigmoid'))
    model.add(keras.layers.Dense(units=500, input_dim=1000, activation='relu'))
    model.add(keras.layers.Dense(units=2, input_dim=500,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
   
#compiling the model
finalmodel = modelbuild()
finalmodel.fit(trainx, trainy, epochs=200, batch_size=120,validation_data=(validx,validy))
























