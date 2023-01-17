

import numpy as np



from keras.datasets import reuters

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.preprocessing.text import Tokenizer



import matplotlib.pyplot as plt

%matplotlib inline
(XTrain, YTrain),(XTest, YTest) = reuters.load_data(num_words=None, test_split=0.3)



print('XTrain class = ',type(XTrain))

print('YTrain class = ',type(YTrain))

print('XTest shape = ',type(XTest))

print('YTest shape = ',type(YTest))



print('XTrain shape = ',XTrain.shape)

print('XTest shape = ',XTest.shape)

print('YTrain shape = ',YTrain.shape)

print('YTest shape = ',YTest.shape)

print('YTrain values = ',np.unique(YTrain))

print('YTest values = ',np.unique(YTest))



unique, counts = np.unique(YTrain, return_counts=True)

print('YTrain distribution = ',dict(zip(unique, counts)))

unique, counts = np.unique(YTest, return_counts=True)

print('YTrain distribution = ',dict(zip(unique, counts)))

plt.figure(1)

plt.subplot(121)

plt.hist(YTrain, bins='auto')

plt.xlabel("Classes")

plt.ylabel("Number of occurrences")

plt.title("YTrain data")



plt.subplot(122)

plt.hist(YTest, bins='auto')

plt.xlabel("Classes")

plt.ylabel("Number of occurrences")

plt.title("YTest data")

plt.show()



print(XTrain[1])

len(XTrain[1])

#The dataset_reuters_word_index() function returns a list where the names are words and the values are integer

WordIndex = reuters.get_word_index(path="reuters_word_index.json")



print(len(WordIndex))



IndexToWord = {}

for key, value in WordIndex.items():

    IndexToWord[value] = key



print(' '.join([IndexToWord[x] for x in XTrain[1]]))

print(YTrain[1])



MaxWords = 10000



# Tokenization of words.

Tok = Tokenizer(num_words=MaxWords)

XTrain = Tok.sequences_to_matrix(XTrain, mode='binary')

XTest = Tok.sequences_to_matrix(XTest, mode='binary')



# Preprocessing of labels

NumClasses = max(YTrain) + 1

YTrain = to_categorical(YTrain, NumClasses)

YTest = to_categorical(YTest, NumClasses)



print(XTrain[1])

print(len(XTrain[1]))
model = Sequential()

model.add(Dense(512, input_shape=(MaxWords,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(NumClasses))

model.add(Activation('softmax'))

model.summary()



model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

history = model.fit(XTrain, YTrain, 

                    validation_data=(XTest, YTest), 

                    epochs=10, 

                    batch_size=64)
Scores = model.evaluate(XTest, YTest, verbose=1)

print('Test loss:', Scores[0])

print('Test accuracy:', Scores[1])
def plotmodelhistory(history): 

    fig, axs = plt.subplots(1,2,figsize=(15,5)) 

    # summarize history for accuracy

    axs[0].plot(history.history['accuracy']) 

    axs[0].plot(history.history['val_accuracy']) 

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy') 

    axs[0].set_xlabel('Epoch')

    axs[0].legend(['train', 'validate'], loc='upper left')

    # summarize history for loss

    axs[1].plot(history.history['loss']) 

    axs[1].plot(history.history['val_loss']) 

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss') 

    axs[1].set_xlabel('Epoch')

    axs[1].legend(['train', 'validate'], loc='upper left')

    plt.show()



# list all data in history

print(history.history.keys())



plotmodelhistory(history)