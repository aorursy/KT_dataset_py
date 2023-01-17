import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
#We see the contents of the root directory
print("Root directory contains: ")
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import tensorflow as tf
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import optimizers
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout, Activation, BatchNormalization

#Declaring the path to the train and test dat
train_path = '../input/train/train'
test_path = '../input/test1/test1'
#Initialize two lists for the data and labels respectively
label=[]
data=[]

#Loop iterating over each file in the training folder
for file in tqdm(os.listdir(train_path)):
    #Reading every image and converting it to grayscale
    image=cv2.imread(os.path.join(train_path,file), cv2.IMREAD_GRAYSCALE)
    #Resizing the image into a manageable size
    image=cv2.resize(image,(96,96))
    #If a file name starts with "cat"
    if file.startswith("cat"):
        label.append(0)
    elif file.startswith("dog"):
        label.append(1)
    try:
        data.append(image/255) 
    except:
        label=label[:len(label)-1]
#Converting our data and labels into numpy arrays
train_data=np.array(data)
train_labels=np.array(label)

print (train_data.shape)
print (train_labels.shape)
#Displaying the first image along with the class it belongs to

plt.imshow(train_data[0], cmap='gray')
plt.title('Class '+ str(train_labels[0]))
#Reshaping our data from a 96x96 array into a 96,96,1 array
train_data = train_data.reshape((train_data.shape)[0],(train_data.shape)[1],(train_data.shape)[2],1)
print(train_data.shape)
print(train_labels.shape)
#Creating the model
model = Sequential()
input_shape = (96,96,1)
model.add(Conv2D(kernel_size=(3,3),filters=32,input_shape=input_shape,activation="relu"))
model.add(Conv2D(kernel_size=(3,3),filters=64,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(5,5),strides=(2,2)))

model.add(Conv2D(kernel_size=(3,3),filters=10,activation="relu"))
model.add(Conv2D(kernel_size=(3,3),filters=5,activation="relu"))

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))

model.add(Flatten())

model.add(Dropout(0.3))

model.add(Dense(100,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))

model.summary()
#We use the ADADELTA optimization on the binary crossentropy loss function
model.compile(optimizer="adadelta",loss="binary_crossentropy",metrics=["accuracy"])
# We try out the model on the training data.
# Train data has been split. 25% of the training data has been kept aside for Validation. 
# We run the fit function for 20 epochs

model_history = model.fit(train_data,train_labels,validation_split=0.25,epochs=20,batch_size=10)
#Visualizing accuracy and loss of training the model
history_dict=model_history.history

#Test Accuracy
train_acc = history_dict['acc']
#Validation Accuracy
val_acc = history_dict['val_acc']

epochs =range(1,len(train_acc)+1)
#Plottig the training and validation loss
plt.plot(epochs, val_acc, 'bo', label='Validation Accuracy')
plt.plot(epochs, train_acc, 'b', label='Train Accuracy')
plt.title('Train and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#Training loss
train_loss = history_dict['loss']
#Validation Loss
val_loss = history_dict['val_loss']

epochs =range(1,len(train_loss)+1)
#Plottig the training and validation loss
plt.plot(epochs, val_loss, 'bo', label='Validation Loss')
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
test_data=[]
id=[]
for file in tqdm(os.listdir(test_path)):
    image_data=cv2.imread(os.path.join(test_path,file), cv2.IMREAD_GRAYSCALE)
    try:
        image_data=cv2.resize(image_data,(96,96))
        test_data.append(image_data/255)
        id.append((file.split("."))[0])
    except:
        print("")
test_data1=np.array(test_data)
print (test_data1.shape)
test_data1=test_data1.reshape((test_data1.shape)[0],(test_data1.shape)[1],(test_data1.shape)[2],1)
dataframe_output=pd.DataFrame({"id":id})
predicted_labels=model.predict(test_data1)
predicted_labels=np.round(predicted_labels,decimals=2)
print(predicted_labels)
labels=[1 if value>0.5 else 0 for value in predicted_labels]
dataframe_output["label"]=labels
dataframe_output.to_csv("submission.csv",index=False)