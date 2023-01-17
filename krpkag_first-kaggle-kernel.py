import os
import numpy as np 
import pandas as pd
import keras
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Conv2D,Convolution2D,  MaxPooling2D, ZeroPadding2D,Dense, Dropout, Activation, Flatten,BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def processData(X,Y):
    test_size = 0.2
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    return X,X_test,Y,Y_test

def getData(path):
    label = os.listdir(path)[0]
    X = np.load(path+ "/"+label+"/X.npy")
    Y = np.load(path+ "/"+label+"/Y.npy")
    print("Loaded Data from npy files")
    return X,Y
X, Y = getData("../input")
plt.imshow(X[5])
plt.show()
X, X_test, Y, Y_test = processData(X,Y)
print ("X Shape : "+ str(X.shape))
print ("Y Shape : "+ str(Y.shape))
# model variable definitions
act = 'sigmoid'
batch_size = 75
filter_pixel=6
droprate=0.25
l2_lambda = 0.0001
reg = l2(l2_lambda) #regularization to use within layers for overfitting
epochs = 100

num_classes = Y.shape[1]
input_shape = X.shape[1:]
model = Sequential()

#convolution 1st layer
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel),
                 activation=act,
                 kernel_regularizer=reg,
                 input_shape=(input_shape)))
model.add(BatchNormalization())
model.add(Dropout(droprate))
model.add(MaxPooling2D())

#convolution 2nd layer
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), kernel_regularizer=reg,activation=act))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(droprate))

#convolution 3rd layer
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel),kernel_regularizer=reg, activation=act))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(droprate))

#Fully connected 1st layer
model.add(Flatten())
model.add(Dense(218,kernel_regularizer=reg))
model.add(BatchNormalization())
model.add(Activation(act))
model.add(Dropout(droprate))

#Fully connected final layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

callbacks = [keras.callbacks.EarlyStopping(patience=10,verbose=1)]
history = model.fit(X, Y,validation_split = 0.2,batch_size=batch_size,epochs=epochs,verbose=1)
print("Finished Fitting Model")
scores = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')