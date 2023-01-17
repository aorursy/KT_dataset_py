import cv2 #Used for Images
import numpy as np #contains a multi-dimentional array and matrix data structures, mathematical operations
from keras.datasets import mnist #Loads the MNIST dataset
from keras.layers import Dense, Flatten #Regular densely-connected NN layer; Flattens the input
from keras.layers.convolutional import Conv2D #2D convolution layer
from keras.models import Sequential #Sequential class which is a linear stack of Layers
from keras.utils import to_categorical #Converts a class vector (integers) to binary class matrix.
import matplotlib.pyplot as plt #creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels
#Loads MNIST dataset and divides the data in train and test.
(X_train, y_train), (X_test, y_test) = mnist.load_data() #
#Displaying any one image from the dataset in gray scale.
plt.imshow(X_train[0], cmap="gray")
plt.show()
print (y_train[0])
# Checking the shape of MNIST dataset.
print ("Shape of X_train: {}".format(X_train.shape))
print ("Shape of y_train: {}".format(y_train.shape))
print ("Shape of X_test: {}".format(X_test.shape))
print ("Shape of y_test: {}".format(y_test.shape))
#Reshaping the data for the model.
#In the output 60000 represents training data; 28,28 represents the image size and 1 represents that it is a gray scale image.
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
#Checking the shape of the training and test data according to the model.
print ("Shape of X_train: {}".format(X_train.shape))
print ("Shape of y_train: {}".format(y_train.shape))
print ("Shape of X_test: {}".format(X_test.shape))
print ("Shape of y_test: {}".format(y_test.shape))
#Converts a class vector (integers) to binary class matrix.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# Declare the Sequential model
model = Sequential([
                    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
                    Conv2D(64, kernel_size=3, activation='relu'),
                    Flatten(),
                    Dense(10, activation='softmax')
]
)
#Display the above model
model.summary()
#Compiling the model.
#adam optimizer is used for learning rate
#categorical_crossentropy loss is used to compute the crossentropy loss between the labels and predictions.
#accuracy metrics calculates how often predictions equals labels.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Training the model for the test validation data based on trained data 
# epochs is the number of times all of the training vectors are used once to update the weights which is set to 3
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
#Predicting the model by giving any random input to the train data
example = X_train[8542]

#Predicts the probability of the input
output_predict = model.predict(example.reshape(1, 28, 28, 1))
print ("Softmax from the neural network:\n {}".format(output_predict))

#Return a new array of given shape and type, filled with zeros.
zers_pred = np.zeros(output_predict.shape)
zers_pred[0][np.argmax(output_predict)] = 1
print ("\nOutput to convert the output to zero and 1: \n {}".format(zers_pred))

print ("\nPrediction of below image:")
plt.imshow(example.reshape(28, 28), cmap="gray")
plt.show()
print("Final Output of the above image: {}".format(np.argmax(output_predict)))