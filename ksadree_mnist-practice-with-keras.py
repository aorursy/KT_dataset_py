#MNIST practice; 
#Francois Chollet's Deep Learning with python (chapters 2 & 5) tutorial
#Visualize activations from hidden layers. 
#MNIST from Keras ; This code gave error. I downloaded Keras MNIST locally and uploaded here
# from keras.datasets import mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#Kaggle MNIST
import numpy as np
import pandas as pd

train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

#Keras MNIST
train_images = np.load('../input/kerasmnist-data/train_images.npy')
train_labels = np.load('../input/kerasmnist-data/train_labels.npy')
test_images = np.load('../input/kerasmnist-data/test_images.npy')
test_labels = np.load('../input/kerasmnist-data/test_labels.npy')
print("train from Keras: " , train_images.shape)
print("train from Kaggle: " , train.shape)
print("test from Keras: " , test_images.shape)
print("test from Kaggle: ", test.shape)
train.head()
print("train labels from Keras:" , train_labels.shape)
print("test labels from Keras: " , test_labels.shape)
train.columns
#The first ten train labels from Keras
train_labels[:10,]
#Extract training labels of Kaggle dataset. Drop the label column.
train_labs = train["label"]
train = train.drop(columns=["label"])
train_labs.shape
type(train)
#Chapter2: Building a fully connected model
#1. Normalize train & test data:
train = train.astype('float32') / 255
test = test.astype('float32') / 255
#2. Build model   "The network architecture"
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation = 'softmax'))
#3. "Compile" choose optimizer, loss function, and metrics to monitor training and testing
network.compile(optimizer='rmsprop' , 
                loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
#4. "Preparing the image data"
#We do not need to reshape training, because it is already in (42000,28*28) shape; Also we normalized in step 1. 
#However we will use Keras test set to monitor training.

#reshape test
test_images = test_images.reshape((10000, 28*28))

#Normalize test
test_images = test_images.astype('float32') / 255
test_images.shape
#5. "Preparing the labels"
from keras.utils import to_categorical 

train_labs = to_categorical(train_labs)
test_labels = to_categorical(test_labels)
#First 5 rows of labels in one-hot encoding
train_labs[:5, :]
#6. Train "network"  using the fit method. We "fit" the model/network to its training data
network.fit(train, train_labs, epochs=5, batch_size=128)
#7. Note the accuracy of 98.70% on the training set. Let's check how the model performs well on the test set:
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test accuracy:  " , test_acc)
#Accuracy on the test set is ~98.05%. 
#The difference with training accuracy (98.70%) represents  overfitting, discussed in chapter 3.
#Chapter 5: Building a convnet model
#1. Instantiate a small convnet
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
#Notes:
#a. Output of a conv2d and maxpool is a 3d tensor
#b. The output shape decreases as we go deeper into the layers
#c. The number of channels is defined by the first argument passed to each conv2d
#d. maxpool2d does not have channel argument.  
#2. Add a classifier on top of the convnet
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
#3. Training the convnet on MNIST images; first reshape
train = train.values.reshape(train.shape[0] , 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0] , 28, 28, 1)

model.compile(optimizer='rmsprop' , 
                loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

model.fit(train, train_labs, epochs=5, batch_size=64)
#View random images from train data:
import matplotlib.pyplot as plt
import numpy as np

def plot2 (i,j):

    plt.subplot(1,2,1)
    plt.imshow(train[i][:,:,0] , cmap=plt.get_cmap('gray'))
    plt.title(np.argmax(train_labs[i]))

    plt.subplot(1,2,2)
    plt.imshow(train[j][:,:,0] , cmap=plt.get_cmap('gray'))
    plt.title(np.argmax(train_labs[j]))

    plt.show()
i, j = np.random.randint(0, train.shape[0]), np.random.randint(0, train.shape[0])
plot2(i,j)
#4. Evaluate model on the test data:
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test accuracy:  " , test_acc)
#~99% accuracy: compare with 98% using a fully connected model. Conv2d model performs significantly better. 
#"Visualizing intermediate activations" (section 5.4). The idea is to capture features in test images that intermdiate
#convnets see. We'll use our convnet "model" above. 
from keras.models import load_model
model.summary()
#Display a random Keras test image:
i = np.random.randint(0, test_images.shape[0])

plt.imshow(test_images[i][:,:,0] , cmap=plt.get_cmap('gray'))
plt.title(np.argmax(test_labels[i]))
plt.show()
#"Instantiating a model from an input tensor and a list of output tensors"
from keras import models
layer_outputs = [layer.output for layer in model.layers[:3]]   #Extracts the output of the top three layers (?)

#creates a model that wil will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#Running the model in predict mode
img_tensor = np.expand_dims(test_images[i], axis=0)       
activations = activation_model.predict(img_tensor)
len(activations)
activations[0].shape
#Activations in the first convolution layer
first_layer_activation = activations[0]
#Plot two channels of the first layer:

plt.subplot(1,2,1)
plt.imshow(first_layer_activation[0, :, :, 5], cmap='gray')

plt.subplot(1,2,2)
plt.imshow(first_layer_activation[0, :, :, 9], cmap='gray')

plt.show()
#Activations in the second convolution layer
second_layer_activation = activations[1]
second_layer_activation.shape
plt.subplot(1,2,1)
plt.imshow(second_layer_activation[0, :, :, 2], cmap='gray')

plt.subplot(1,2,2)
plt.imshow(second_layer_activation[0, :, :, 30], cmap='gray')

plt.show()
#Activations in the third convolution layer
third_layer_activation = activations[2]
third_layer_activation.shape
plt.subplot(1,2,1)
plt.imshow(third_layer_activation[0, :, :, 2] , cmap = 'gray')

plt.subplot(1,2,2)
plt.imshow(third_layer_activation[0, :, :, 50] , cmap = 'gray')

plt.show()
#Note: The activations are harder to interpret deeper into the network. Higher dimension layers capture 
#more detailed information from the image. Lower dimension layers resemble the image as they capture an overall shape
#of the input. 
#Submission
test = test.values.reshape(test.shape[0], 28, 28, 1)
predictions = model.predict_classes(test)
submission = pd.DataFrame({"ImageID" : list(range(1,len(predictions)+1)) , "Label" : predictions})
submission.to_csv("submission.csv" , header=True, index=False)