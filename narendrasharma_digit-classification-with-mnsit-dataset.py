from IPython.display import Image
Image("../input/cnnimag1/CNN_Image1.png")
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from  tensorflow.keras.utils import to_categorical,plot_model 
import tensorflow as tf
## Sometimes it takes Longer Just to Confirm all is done . 
print("Import Done")
mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
(xtrain, ytrain),(xtest, ytest) = mnist.load_data() # Load data
## Lets Look at the Shape of Train and Test Dataset 
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)
idx=2
plt.imshow(xtrain[idx], cmap="gray") # Import the image
print(ytrain[idx])
plt.show() # Plot the image
fig,axes= plt.subplots(nrows=4,ncols=5,figsize=(12,6))
axes=axes.flatten()
for i,ax in zip(range(20),axes):
    ax.imshow(xtrain[i],cmap='gray')
    ax.set_title(ytrain[i])
    ax.axis('off')
plt.show()
print(xtrain.shape)
print(xtest.shape)
xtrain=xtrain.reshape(60000,28,28,1)
xtest=xtest.reshape(10000,28,28,1)
print(xtrain.shape)
print(xtest.shape)
# Scaling down Data
xtrain=xtrain/255
xtest=xtest/255
## Idx is just a value can be changed from any value between 0 -59999 there are total 60,000 Image 
idx=10
plt.imshow(xtrain[idx].reshape(28,28), cmap="gray") # Import the image
plt.title(ytrain[idx])  ## Put the Title of the Image . 
plt.show() # Plot the image
# Printing the ytest and ytrain output labels
print(ytest)
print(ytrain)
## Printing Shape before and After the One Hot Encoding
print(ytrain.shape)
print(ytest.shape)
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)
print(ytrain.shape)
print(ytest.shape)
# Printing the labels for first 10 columns after one hot encoding . 
ytrain[0:10]
from tensorflow.keras import models,layers
# Create Sequential Models
model=models.Sequential()
## Add the Layers for Convocalation 
model.add(layers.Conv2D(filters=10,kernel_size=(2,2),input_shape=(28,28,1),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

## Add Second Conventional Layer to Model
model.add(layers.Conv2D(filters=12,kernel_size=(2,2),activation='relu'))
model.add(layers.Conv2D(filters=20,kernel_size=(2,2),activation='relu'))

## Adding Max Pooling Layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

## Flattne Layer
model.add(layers.Flatten())

### Classification Segemention to the  
model.add(layers.Dense(150,activation='relu'))
model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dense(50,activation='relu'))

########Output layer for 
model.add(layers.Dense(10,activation='softmax'))


## Printing the Model 
plot_model(model)
## Printing the Summary of Model 
model.summary()
# Model Compilation

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
# Here we are fitting 
history=model.fit(xtrain, ytrain, epochs=20,batch_size=1000,verbose=True,validation_data=(xtest, ytest))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.title("Showing the Train Vs Test Accuracy")
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(xtest,  ytest, verbose=2)