import numpy as np

import pandas as pd

import random as rn

import tensorflow as tf

from keras import backend as K



session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)





rn.seed(20)

np.random.seed(8)

tf.set_random_seed(1998)
train = pd.read_csv("../input/train.csv")

print(train.shape)

print(train.head())
test = pd.read_csv("../input/test.csv")

print(test.shape)

print(test.head())
XTrain = train.iloc[:,1:].values.astype('float32')

yTrain = train.iloc[:,0].values.astype('int32')



XTest = test.values.astype('float32')
#Converting Flattened images to its 2D matrix form

XTrain = XTrain.reshape(XTrain.shape[0], 28, 28)

XTest = XTest.reshape(XTest.shape[0], 28, 28)
import matplotlib.pyplot as plt



for i in range(6):

    plt.subplot(231 + i) #2 row, 3 col -> 6 cells; place image in 1st, 2nd,..6th cell

    plt.imshow(XTrain[100+i])

    plt.title(yTrain[100+i])

#Expand one more dimension for Colour channel Gray

XTrain = XTrain.reshape(XTrain.shape[0], 28, 28, 1)

XTest = XTest.reshape(XTest.shape[0], 28, 28, 1)



#Normalise inputs from 0-255 to 0-1

XTrain = XTrain/255

XTest = XTest/255
#OneHot Encoding of labels

from keras.utils.np_utils import to_categorical

yTrain = to_categorical(yTrain)



#Plotting one of the label

import matplotlib.pyplot as plt

plt.figure()

plt.title(yTrain[1000])

plt.plot(yTrain[1000])

plt.xticks(range(10))
#import libraries

from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
def convNeuralNetwork():

    model = Sequential([

        Convolution2D(32, input_shape=(28,28,1), kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),

        MaxPooling2D(pool_size=(2,2), strides=(2,2)),

        BatchNormalization(),

        Dropout(0.3),

        Convolution2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),

        MaxPooling2D(pool_size=(2,2), strides=(2,2)),

        BatchNormalization(),

        Dropout(0.3),

        Convolution2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),

        MaxPooling2D(pool_size=(2,2), strides=(2,2)),

        BatchNormalization(),

        Dropout(0.3),

        Flatten(),

        Dense(96, activation='relu'),

        Dense(10, activation='softmax')

    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

    
classifier = convNeuralNetwork()

classifier.optimizer.lr = 0.01

print(classifier.summary())
#Fit Model

history = classifier.fit(XTrain, yTrain, batch_size=10, epochs=20, validation_split=0.2,  verbose=1)
from sklearn.metrics import confusion_matrix, accuracy_score

yPred = classifier.predict(XTrain)

print(yPred[0])

cm = confusion_matrix(np.argmax(yTrain,axis=1),np.argmax(yPred,axis=1))

print("Confusion Matrix\n:",cm)

accuracy = accuracy_score(np.argmax(yTrain,axis=1),np.argmax(yPred,axis=1))

print("Accuracy on dataset = ",accuracy)



from matplotlib import pyplot as plt

plt.figure()

plt.plot(history.history['acc'],'green',label='Training Accuracy')

plt.plot(history.history['val_acc'],'blue',label='Validation Accuracy')

plt.plot(history.history['loss'],'red',label='Loss')

plt.plot(history.history['val_loss'],'orange',label='Validation Loss')

plt.title('Validation Accuracy & Loss')

plt.xlabel('Epoch')

predictions = classifier.predict_classes(XTest, verbose=1)

result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

result.to_csv("result4.csv", index=False, header=True)
#Serialize model to JSON

model_json = classifier.to_json()

with open("model4.json", "w") as json_file:

    json_file.write(model_json)



#Serialize weights to HDF5

classifier.save_weights("model4.h5")