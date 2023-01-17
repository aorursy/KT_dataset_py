import keras

import seaborn as sns

import numpy as np

import pandas as pd

from keras.utils import to_categorical

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten

from keras.optimizers import Adam

from keras.models import Sequential

from matplotlib import pyplot as plt



train=pd.read_csv("../input/digit-recognizer/train.csv")

#The data is read from the second row onwards , Since the first row contains headers [label,pixel0,pixel1,...]

trainX=train.iloc[1:,1:]

trainY=train.iloc[1:,0]



#Determining output labels

labels=np.unique(list(trainY))

num_class=len(labels)



#Convert the pd dataframes into numpy arrays

trainX=np.array(trainX)

trainY=np.array(trainY)



#Deciding size of the Validation and Test sets

val_size=int(len(trainX)*0.1)

test_size=int(len(trainX)*0.2)



#Validation data preparation

valX=trainX[:val_size].copy()

valY=trainY[:val_size].copy()

trainX=trainX[val_size:]

trainY=trainY[val_size:]



#Test Data Preparation

testX=trainX[:test_size].copy()

testY=trainY[:test_size].copy()

trainX=trainX[test_size:]

trainY=trainY[test_size:]



#Reshaping the row pixel values into a matrix where each matrix constitutes an image of a handwritten digit 

trainX=trainX.reshape((-1,28,28,1))

valX=valX.reshape((-1,28,28,1))

testX=testX.reshape((-1,28,28,1))



#Since the output variable is to categorical , I'm converting them into One Hot Encoded Form. This will facilitate is outputting 

#the predictions in human-understanble form(Will become clear during the prediction.)

trainY=to_categorical(trainY,num_classes=10)

testY=to_categorical(testY,num_classes=10)

valY=to_categorical(valY,num_classes=10)

#Initialising an instance of class Sequential to hold the model layers

model=Sequential()



##Starting to add the model layers



#Adding the first Conv layer with a large kernel size of (7,7) to capture the large features 

model.add(Conv2D(8,(7,7),input_shape=(28,28,1),activation='relu',padding='same'))

#Adding MaxPooling to reduce the dimensionality of feature map to contain the dominant features

model.add(MaxPooling2D((2,2)))



#Two more consecutive Conv layers are added to deepen the networks capturing better

#representational features of the data

model.add(Conv2D(32,(5,5),activation='relu',padding='same'))

model.add(MaxPooling2D((2,2)))



model.add(Conv2D(32,(5,5),activation='relu',padding='same'))

model.add(MaxPooling2D((2,2)))



#Fully connected layers to process feature maps from Conv layers

model.add(Dense(128,activation='relu'))

model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(16,activation='relu'))

#Final fully connected layer to arrive at class probabilities across 10 classes (Digits 0 - 9)

model.add(Dense(10,activation='softmax'))













#Adam optimiser with learning rate = 0.001

opt=Adam(lr=1e-3)

#Compiling model. Loss is Categorical crossentropy since we are predicting across more than two classes

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#Fitting model on training data and the learned weights on train data is used to make predictions on validation data

#for making sure that model is generalising(Will become clear while plotting Accuracy v/s Epochs graph)

H=model.fit(x=trainX,y=trainY,validation_data=(valX,valY),epochs=20,batch_size=32)







#History object

H.history




plt.style.use("ggplot")

plt.figure()

#Since I'm are training for only 20 epochs the the np.arange plots values from 0 to 20 in x-axis

#With the Accuracy in y-axis

#Training accuracy

plt.plot(np.arange(0,20),H.history['accuracy'],label="train_acc")

#Validation accuracy

plt.plot(np.arange(0,20),H.history['val_accuracy'],label="val_acc")

plt.title("Accuracy v/s Epochs")

plt.xlabel("Epoch#")

plt.ylabel("Accuracy")

plt.legend()

plt.show()



plt.style.use("ggplot")

plt.figure()

#Same as Accuracy plotting 

#However here loss in the y-axis

#Training loss

plt.plot(np.arange(0,20),H.history['loss'],label="train_loss")

#Validation loss

plt.plot(np.arange(0,20),H.history['val_loss'],label="val_loss")

plt.title("Loss v/s Epochs")

plt.xlabel("Epoch#")

plt.ylabel("Loss")

plt.legend()

plt.show()



import pandas as pd

test=pd.read_csv("../input/digit-recognizer/test.csv")
testX=test.iloc[0:,:]
testX=np.array(testX)

testX=testX.reshape((-1,28,28,1))

pred=model.predict(testX)
pred=np.argmax(pred,axis=1)
pred.shape
test_result=pd.Series(pred,name="Label")
test_result
table=pd.concat([pd.Series(range(1,28001),name="ImageId"),test_result],axis=1)
table

table=table.to_csv("Test results.csv",index=False)