# I am a beginner in machine learning and data analysis, I hope to be promoted here, thanks.
#This example uses SGD as an optimization function and the accuracy is not very high.

#packages loda in

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from keras.utils.np_utils import to_categorical

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
#input data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#prepare data

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

del train

X_train = X_train.astype('float32') / 255.0

X_test = test.astype('float32') / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

X_test = test.values.reshape(-1,28,28,1)
#use keras for one-hot coding

n_classes = 10

Y_train = to_categorical(Y_train, n_classes)
#Evaluate with 10% data

np.random.seed(2)

X_train, X_values, Y_train, Y_values = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)
#Relying on Sequential to define a forward neural network

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'Same', 

                 activation ='relu', input_shape = (28, 28, 1)))

model.add(Conv2D(filters = 32, kernel_size = (3, 3),padding = 'Same', 

                 activation ='relu'))

#Pool the activation step and add a Dropout layer

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#Will flatten the model and pass the result to the Softmax function

model.add(Flatten())

model.add(Dense(260, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
#Use the SGD function

optimizer = SGD(lr=0.045, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
#Fitting model

history = model.fit(X_train, Y_train, batch_size = 84, epochs = 5, 

          validation_data = (X_values, Y_values), verbose = 1)
#Assess classification results

model.evaluate(X_values, Y_values, verbose = 0)
#Observe the training prediction results and loss function

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='g', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='g',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
#Forecast the contents of the Test file

predict_results = model.predict(X_test)

predict_results = np.argmax(predict_results,axis = 1)

predict_results = pd.Series(predict_results,name="Label")

results = pd.concat([pd.Series(range(1,28001),name = "Image_ID"),predict_results],axis = 1)

results.to_csv("cnn_mnist_predict_results.csv",index=False)