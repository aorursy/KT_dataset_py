# keras imports
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, Adadelta
from keras.models import load_model
from keras.regularizers import l2

# Generic imports
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np, string, pickle, warnings, random
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
topWords = 50000
MAX_LENGTH = 200
nb_classes = 2

# Downloading data
imdbData = imdb.load_data(path='imdb.npz', num_words=topWords)

(x_train, y_train), (x_test, y_test) = imdbData
stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", \
             "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", \
             'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', \
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', \
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
             'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', \
             'above', 'below', 'to', 'from', 'off', 'over', 'then', 'here', 'there', 'when', 'where', 'why', \
             'how', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'own', 'same', 'so', \
             'than', 'too', 's', 't', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
             've', 'y', 'ma']
word2Index = imdb.get_word_index()
index2Word = {v: k for k, v in word2Index.items()}
index2Word[0] = ""
sentimentDict = {0: 'Negative', 1: 'Positive'}

def getWordsFromIndexList(indexList):
    wordList = []
    for index in indexList:
      if index in index2Word:
        wordList.append(index2Word[index])

    return " ".join(wordList)

def getSentiment(predictArray):
    pred = int(predictArray[0])
    return sentimentDict[pred]

def getIndexFromWordList(wordList):
    indexList = []
    for word in wordList:
      if word in word2Index:
        indexList.append(str(word2Index[word]))
        
    return indexList
print (len(word2Index))
print(getWordsFromIndexList(x_train[0]))
print(len(x_train[0]), x_train[0])
stopIndexList = []

for stopWord in stopWords:
    stopIndexList.append(word2Index[stopWord])

trainData = []

for indexList in x_train:
    processedList = [index for index in indexList if index not in stopIndexList]
    trainData.append(processedList)
    
x_train = trainData
'''
Padding data to keep vectors of same size
If size < 200 then it will be padded, else it will be cropped
'''
trainX = pad_sequences(x_train, maxlen = MAX_LENGTH, padding='post', value = 0.)
testX = pad_sequences(x_test, maxlen = MAX_LENGTH, padding='post', value = 0.)

'''
One-hot encoding for the classes
'''
trainY = np_utils.to_categorical(y_train, num_classes = nb_classes)
testY = np_utils.to_categorical(y_test, num_classes = nb_classes)
print(len(trainX[0]), trainX[0])
sgdOptimizer = 'adam'
lossFun='categorical_crossentropy'
batchSize=1024
numEpochs = 50
numHiddenNodes = 128
EMBEDDING_SIZE = 300
denseLayer1Size = 256
denseLayer2Size = 128
model = Sequential()

# Train Embedding layer with Embedding Size = 300
model.add(Embedding(topWords, EMBEDDING_SIZE, input_length=MAX_LENGTH, mask_zero=True, name='embedding_layer'))

# Define Deep Learning layer
model.add(Bidirectional(LSTM(numHiddenNodes), merge_mode='concat',name='bidi_lstm_layer'))

# Define Dense layers
model.add(Dense(denseLayer1Size, activation='relu', name='dense_1'))
model.add(Dropout(0.25, name = 'dropout'))
model.add(Dense(denseLayer2Size, activation='relu', name='dense_2'))

# Define Output Layer
model.add(Dense(nb_classes, activation='softmax', name='output'))

model.compile(loss=lossFun, optimizer=sgdOptimizer, metrics=["accuracy"])
print(model.summary())
model.fit(trainX, trainY, batch_size=batchSize, epochs=numEpochs, verbose=1, validation_data=(testX, testY))
score = model.evaluate(testX, testY, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
predY = model.predict_classes(testX)
yPred = np_utils.to_categorical(predY, num_classes = nb_classes)
print("Classification Report:\n")
print(classification_report(testY, yPred))
model.save('imdb_bi_lstm_tensorflow_model.hdf5')
loaded_model = load_model('imdb_bi_lstm_tensorflow_model.hdf5')
print(loaded_model.summary())
num = 121
num_next = num + 1
print("Testing for test case..." + str(num))
groundTruth = testY[num]

sampleX = testX[num:num_next]
predictionClass = loaded_model.predict_classes(sampleX, verbose=0)
prediction = np_utils.to_categorical(predictionClass, num_classes = nb_classes)[0]

print("Text: " + str(getWordsFromIndexList(x_test[num-1])))
print("\nPrediction: " + str(getSentiment(predictionClass)))
if np.array_equal(groundTruth,prediction):
    print("\nPrediction is Correct")
else:
    print("\nPrediction is Incorrect")