from keras.layers import Input, Dense

from keras.models import Model

from keras.datasets import mnist



import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline
#Load our MNIST dataset.

(XTrain, YTrain), (XTest, YTest) = mnist.load_data()



print('XTrain class = ',type(XTrain))

print('YTrain class = ',type(YTrain))



# shape of our dataset.

print('XTrain shape = ',XTrain.shape)

print('XTest shape = ',XTest.shape)

print('YTrain shape = ',YTrain.shape)

print('YTest shape = ',YTest.shape)

# Number of distinct values of our MNIST target

print('YTrain values = ',np.unique(YTrain))

print('YTest values = ',np.unique(YTest))

# Distribution of classes in our dataset.

unique, counts = np.unique(YTrain, return_counts=True)

print('YTrain distribution = ',dict(zip(unique, counts)))

unique, counts = np.unique(YTest, return_counts=True)

print('YTest distribution = ',dict(zip(unique, counts)))

# we plot an histogram distribution of our test and train data.

fig, axs = plt.subplots(1,2,figsize=(15,5)) 

axs[0].hist(YTrain, ec='black')

axs[0].set_title('YTrain data')

axs[0].set_xlabel('Classes') 

axs[0].set_ylabel('Number of occurrences')

axs[1].hist(YTest, ec='black')

axs[1].set_title('YTest data')

axs[1].set_xlabel('Classes') 

axs[1].set_ylabel('Number of occurrences')

# We want to show all ticks...

axs[0].set_xticks(np.arange(10))

axs[1].set_xticks(np.arange(10))



plt.show()

# Data normalization.

XTrain = XTrain.astype('float32') / 255

XTest = XTest.astype('float32') / 255

# data reshapping.

XTrain = XTrain.reshape((len(XTrain), np.prod(XTrain.shape[1:])))

XTest = XTest.reshape((len(XTest), np.prod(XTest.shape[1:])))



print (XTrain.shape)

print (XTest.shape)

InputModel = Input(shape=(784,))

EncodedLayer = Dense(32, activation='relu')(InputModel)

DecodedLayer = Dense(784, activation='sigmoid')(EncodedLayer)



AutoencoderModel = Model(InputModel, DecodedLayer)

# we can summarize our model.

AutoencoderModel.summary()

# Let's train the model using adadelta optimizer

AutoencoderModel.compile(optimizer='adadelta', loss='binary_crossentropy')



history = AutoencoderModel.fit(XTrain, XTrain,

                    batch_size=256,

                    epochs=100,

                    shuffle=True,

                    validation_data=(XTest, XTest))

# Make prediction to decode the digits

DecodedDigits = AutoencoderModel.predict(XTest)
def plotmodelhistory(history): 

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Autoencoder Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



# list all data in history

print(history.history.keys())

# visualization of the loss minimization during the training process

plotmodelhistory(history)
n=5

plt.figure(figsize=(20, 4))

for i in range(n):

    ax = plt.subplot(2, n, i + 1)

    # input image

    plt.imshow(XTest[i+10].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)

    # Image decoded by our Auto-encoder

    plt.imshow(DecodedDigits[i+10].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()