#Tensorflow and Keras

import tensorflow as tf

from tensorflow import keras



# Import of keras model and hidden layers for our convolutional network

from keras.models import Sequential

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import Dense, Flatten, Dropout



#Image handling libraries

import numpy as np

import matplotlib.pyplot as plt

import cv2

import pandas as pd



#Sklearn libraries

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



#Initialize a list of paths for images

imagepaths = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        #print(os.path.join(dirname, filename))

        if path.endswith("png"):

            imagepaths.append(path)



print(len(imagepaths))

#print(imagepaths)
#Defining a function that plots the image selected from a path



def img_plot(img_path):

    img = cv2.imread(img_path)

    #convert to RGB space

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #check the shape of the image

    print("Shape of the image is ", img_rgb.shape)

    #Display the image

    plt.grid(False)

    plt.imshow(img_rgb)

    plt.xlabel("Width")

    plt.ylabel("Height")

    plt.title("Image " + img_path)
#Example image plot

#Plotting the first image from the dataset

img_plot(imagepaths[0])
#### Creating Training Set and Labels ####

# X for image data

X = []

# y for the labels

y = []



#Load the images into X by doing the necessary conversions and resizing of images

#Resizing is done to reduce the size of image to increase the speed of training

for path in imagepaths[:19999]:

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (128,128))

    X.append(img)  

    #Getting the labels from the image path

    category = path.split("/")[7]

    #print(category)

    label = int(category.split("_")[0][1])

    #print(label)

    y.append(label)



#print(label)

#Turning X & y into numpy arrays

X = np.array(X)

X = X.reshape(len(imagepaths[:19999]), 128, 128, 1)

y = np.array(y)



print("Images loaded: ", len(X))

print("Labels loaded: ", len(y))



print(y[0], imagepaths[0]) #To debug
# Make the test train split

threshold = 0.3



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = threshold, random_state = 42)
# Create a CNN Sequential Model

model = Sequential()

model.add(Conv2D(32, (5,5), activation = 'relu', input_shape=(128,128,1)))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3, 3), activation='relu')) 

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu')) 

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu')) 

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))
#Model configuration for training purpose

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2, 

         validation_data=(X_test, y_test))
model.save('handgesturerecog_model.h5')
#calculate loss and accuracy on test data



tLoss, tAccuracy = model.evaluate(X_test, y_test)



print('Test accuracy: {:2.2f}%'.format(tAccuracy*100))
# Making predictions on test data

prediction = model.predict(X_test)
#Lets compare the predicted value with actual label value

# Ideally both prediction[0] and y_test[0] should be same

np.argmax(prediction[0]), y_test[0]
# Function to plot images and labels for validation purposes

def validate_gestures(predictions_array, true_label_array, img_array):

  # Array for pretty printing and then figure size

  class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"] 

  plt.figure(figsize=(15,5))

  

  for i in range(1, 10):

    # Just assigning variables

    prediction = predictions_array[i]

    true_label = true_label_array[i]

    img = img_array[i]

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    

    # Plot in a good way

    plt.subplot(3,3,i)

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)



    predicted_label = np.argmax(prediction) # Get index of the predicted label from prediction

    

    # Change color of title based on good prediction or not

    if predicted_label == true_label:

      color = 'blue'

    else:

      color = 'red'



    plt.xlabel("Predicted: {} {:2.0f}% (Actual: {})".format(class_names[predicted_label],

                                  100*np.max(prediction),

                                  class_names[true_label]),

                                  color=color)

  plt.show()
# Plot testing based on predictions and their actual values

validate_gestures(prediction, y_test, X_test)
#Transform predictions into 1D array 

y_pred = np.argmax(prediction, axis=1)
#Create a Confusion Matrix for Evaluation

# H = Horizontal

# V = Vertical

pd.DataFrame(confusion_matrix(y_test, y_pred), 

             columns=["Predicted Thumb Down", "Predicted Palm (H)", "Predicted L", "Predicted Fist (H)", "Predicted Fist (V)", "Predicted Thumbs up", "Predicted Index", "Predicted OK", "Predicted Palm (V)", "Predicted C"],

             index=["Actual Thumb Down", "Actual Palm (H)", "Actual L", "Actual Fist (H)", "Actual Fist (V)", "Actual Thumbs up", "Actual Index", "Actual OK", "Actual Palm (V)", "Actual C"])