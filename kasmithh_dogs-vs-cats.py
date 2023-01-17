import numpy as np 
import pandas as pd 
import cv2
import os
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import matplotlib.pyplot as plt

#The following code allows us to refer to the directories where the images are located.
#This will help later when it comes time to load in the images and shape them.
Train_dir = '../input/train'
Test_dir = '../input/test'
IMG_SIZE = 100

#Code that helped with loading in images and processing images can be found here https://www.youtube.com/watch?v=gT4F3HGYXf4&t=554s
#The code below aids in labeling each image as "cat" or "dog"
#The fuction below also labels the images as one-hot arrays
#so that they can be passed through the CNN as output labels.
def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'cat':
        return [1,0]
    elif word_label == 'dog':
        return [0,1]
#The second line of the code above was confusing to me at first so I decided to
#make a print statement that helped me to understand it better.
#Running the print statement below shows that it grabs the text 
#that indicates if the picture is a cat or dog. It splits the image by '.'
#and then takes the 0-ith object in the index which is "cat" or "dog".
words = 'cat.0.jpg'
words2 = words.split('.')[0]
print(words2)
def create_train_data():
    training_data = []
    #os.listdir returns a list of all of the files and folders in the path
    #So we pass the path to the training directory that we created earlier
    #https://www.youtube.com/watch?v=iI2zR1WLPZ8
    for img in os.listdir(Train_dir):
        label = label_img(img)
        #os.path.join seems to create a path to each individual image by
        #combining the path to the training directory and the image name
        #https://docs.python.org/3/library/os.path.html#os.path.join
        path = os.path.join(Train_dir,img)
        #The code below uses cv2 to read in the image in Grayscale and then resize
        #the images to 100x100 which we created earlier with the variable IMG_SIZE
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        #Appends the result of each image to a numpy array so we can work with it later
        training_data.append([np.array(img), np.array(label)])
        shuffle(training_data)
    return training_data

#Run function to get our training data
training_data = create_train_data()
X = np.array([i[0] for i in training_data])
X = X.reshape(-1,100,100,1)
X = X/255
Y = np.array([i[1] for i in training_data])

print(X.shape)
model = Sequential()
model.add(Conv2D(64, kernel_size = (3,3), input_shape=(100,100,1), activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer='adam',
              loss= 'binary_crossentropy',
              metrics=['accuracy', 'mae'])
Fit = model.fit(X, Y, epochs = 5, validation_split = 0.30)
plt.plot(Fit.history['loss'])
plt.plot(Fit.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()
plt.plot(Fit.history['acc'])
plt.plot(Fit.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()