import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.optimizers import *
from keras.layers import *
import gensim
import string
import os
print(os.listdir("../input"))
# Use pandas to read the dataset
dataset = pd.read_csv('../input/articles.csv')
# Convert the dataset to numpy array and gets the text row only
dataset = dataset.values[:,1]

print (dataset.shape)
# Create a translator to map every character in the outlab to inlab
inlab = ''
outlab = string.punctuation + '–'
tranlator = str.maketrans(inlab, inlab, outlab)

# array to add the words
brokePhrases = []

# for every news
for i in range (0, dataset.shape[0]):
  # get the news
  new = str(dataset[i])  
  # for every phrase in that news 
  for phrase in new.split('.')[:-1]:
    # make all characteres lower
    editedPhrase = phrase.lower()
    # put none in every position where a character is found
    editedPhrase = editedPhrase.translate(tranlator)
    # break the phrase in words and add that splited phrase to the list
    brokePhrases.append([w for w in editedPhrase.split(' ') if w != ''])
    
print (brokePhrases[0])
w2v_model = gensim.models.Word2Vec(brokePhrases, size=50, window=7, min_count=10, workers=8)
print(w2v_model.wv.most_similar('13', topn=5))
print(w2v_model.wv.most_similar('deus', topn=5))
print(w2v_model.wv.most_similar('país', topn=5))
print(w2v_model.wv.most_similar('computador', topn=5))
print(w2v_model.wv.most_similar(positive=['rei', 'mulher'], negative=['homem'], topn=5))
print(w2v_model.wv.most_similar(positive=['mãe', 'menino'], negative=['pai'], topn=5))
print(w2v_model.wv.most_similar(positive=['doutor', 'mulher'], negative=['homem'], topn=5))
# get the weights of the model
w2v_weights = w2v_model.wv.vectors
print(w2v_weights.shape)
# size of the sequence to train and predict
sequenceSize = 10
# number of samples for the LSTM dataset 
maxNumberSamples = 1000000

# alloc the dataset
datasetLSTM_X = np.zeros(shape=(maxNumberSamples,sequenceSize, w2v_weights.shape[1]))
datasetLSTM_Y = np.zeros(shape=(maxNumberSamples, w2v_weights.shape[1]))

# Every sample is a sequence of a vector with the size of the model output
print(datasetLSTM_X.shape)
# Every sample is a vector with the size of the model output
print(datasetLSTM_Y.shape)
# get the maxNumberSamples first in the text
for i in range (0, maxNumberSamples):
    # verify if the news has at least sequenceSize number of words
    if len(brokePhrases[i]) > sequenceSize:
        # iterator to index the news              
        for j in range (0, len(brokePhrases[i])-sequenceSize-1):            
            try:
                # gets the sequenceSize words to create X
                datasetLSTM_X[i] = np.array([w2v_model.wv[w] for w in brokePhrases[i][j:j+sequenceSize]])
                # the the next word of the sequenceSize words before
                datasetLSTM_Y[i] = np.array([w2v_model.wv[brokePhrases[i][j+sequenceSize+1]]])
            except KeyError:
                pass
                                  
#split the datasets
train_x = datasetLSTM_X[:int(0.9*maxNumberSamples)]
train_y = datasetLSTM_Y[:int(0.9*maxNumberSamples)]
val_x = datasetLSTM_X[int(0.9*maxNumberSamples):]
val_y = datasetLSTM_Y[int(0.9*maxNumberSamples):]
sampleToShow = 2000
for i in range(0, sequenceSize):
    print(w2v_model.most_similar(train_x[sampleToShow][i].reshape(1,w2v_weights.shape[1]))[0][0], end=" ")

print(w2v_model.most_similar(train_y[sampleToShow].reshape(1,w2v_weights.shape[1]))[0][0])

# creates and compile the model
model = Sequential()
model.add(LSTM(256, input_shape=(sequenceSize,w2v_weights.shape[1])))
model.add(Dense(w2v_weights.shape[1],  activation='linear'))
model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics=['mse'])
# the initial phrase
initialPhrase = ['na', 'última', 'semana', 'o', 'brasil', 'ficou', 'sabendo', 'de', 'uma', 'notícia']

k = 0
while (k < 10):
    # train for one epoch
    history = model.fit(train_x, train_y, epochs=1, batch_size=2048, shuffle=False, validation_data=(val_x, val_y), verbose=1)
    
    if k % 2 == 0:
        i = 0
        # convert the array to wordvec
        array = np.array([w2v_model.wv[initialPhrase[0]], w2v_model.wv[initialPhrase[1]], \
                          w2v_model.wv[initialPhrase[2]], w2v_model.wv[initialPhrase[3]], \
                          w2v_model.wv[initialPhrase[4]], w2v_model.wv[initialPhrase[5]], \
                          w2v_model.wv[initialPhrase[6]], w2v_model.wv[initialPhrase[7]], \
                          w2v_model.wv[initialPhrase[8]], w2v_model.wv[initialPhrase[9]]])

        # print the next ten words predicted
        print(initialPhrase, end=" ")
        
        while(i < 10):
            # gets the net output
            outputLSTM = model.predict(array.reshape(1,sequenceSize,w2v_weights.shape[1]))
            # shift to left the phrase
            array = np.roll(array, -1)
            # and puts the predicted word as the last word of the array to predict
            array[-1] = (w2v_model.wv[(w2v_model.most_similar(outputLSTM)[0][0])])
            # gets the most similar word that represents the predicted word 
            print(w2v_model.most_similar(array[-1].reshape(1,w2v_weights.shape[1]))[0][0], end=" ")

            i += 1

        print("\n")
    k+=1
