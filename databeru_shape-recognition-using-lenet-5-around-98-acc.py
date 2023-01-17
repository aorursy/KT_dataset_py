import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
def rgb_to_gray(rgb):
# Convert rgb images to gray images
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Create a list with the shape of the images (circles, squares or triangles)
# and the images
lst_images = []

directories = os.listdir("../input/basicshapes/shapes")
directories.remove("shapes")

for d in directories:
    for i in os.listdir("../input/basicshapes/shapes/"+d):
        img = plt.imread("../input/basicshapes/shapes/"+d+"/"+i)
        img = rgb_to_gray(img)
        # Reshape the images to 28x28x1 for the neural network
        img = np.array(img).reshape(28,28,1)
        lst_images.append([d,img])
# Shuffle the list to make the training more effective
import random
random.shuffle(lst_images)
# Separate the images from the labels
# Rename the labels into:
# squares => 0
# circles => 1
# triangles => 2
X = []
y = []
for i in range(len(lst_images)):
    X.append(lst_images[i][1])
    
    yi = lst_images[i][0]
    if yi == "squares":
        y.append(0)
    elif yi == "circles":
        y.append(1)
    else:
        y.append(2)
# Show a few images of the dataset with labels
for i in range(3):
    plt.imshow(X[i].reshape(28,28))
    plt.title(lst_images[i][0], fontsize =18)
    plt.show()
# Convert the labels y with to_categorical for the neural network
# Example: [2,1] => [[0,0,1],[0,1,0]]
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# Split the dataset into a train-set and a test-set
from sklearn.model_selection import train_test_split
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
import tensorflow.keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, MaxPooling2D, BatchNormalization, AveragePooling2D
from sklearn.metrics import confusion_matrix, accuracy_score

def from_categorical(lst):
    """
    Inverse of to_categorical
    Example: [[0,0,0,1,0], [1,0,0,0,0]] => [3,0]
    """
    
    lst = lst.tolist()
    lst2 = []
    for x in lst:
        lst2.append(x.index(max(x)))
    return lst2

def LeNet(Conv2D_filters = 128, 
          validation_split = 0.2,
          X_train = X_train, 
          X_test = X_test, 
          y_train = y_train, 
          y_test = y_test):
    
    # Create the LeNet model 
    model = Sequential()
    model.add(Conv2D(filters=Conv2D_filters, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=Conv2D_filters*2, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=3, activation = 'softmax'))
    
    # Compile and train the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x = X_train, y = y_train, batch_size = 128, epochs = 100, verbose = 0, validation_split = validation_split)
    
    # Display the results
    # print("################### New model ###################")
    length = len(model.history.history["accuracy"])
    plt.plot(np.arange(0, length), model.history.history["accuracy"], label="accuracy")
    
    # Display the validation results only if there is a validation split
    if validation_split > 0:
        plt.plot(np.arange(0, length), model.history.history["val_accuracy"], label="val_accuracy")
        plt.title(f"Accuracy & Validation accuracy\nNumber of Conv. filters: {Conv2D_filters}")
    else:
        plt.title(f"Accuracy\nNumber of Conv. filters: {Conv2D_filters}")
        
    plt.xlabel("Epoch #")
    plt.show()

    y_test2 = from_categorical(y_test)
    pred = model.predict_classes(X_test)

    print("### Test-set ###\n\nConfusion Matrix:\n")
    print(confusion_matrix(y_test2,pred))
    print(f"\nAccuracy: {accuracy_score(y_test2,pred)}")
    
    return model
for f in [2**x for x in range(4,8)]:
    LeNet(f)
model = LeNet(64, 0)
# Rotating an image
img = X_train[2]
for i in range(0,4):
    plt.figure(figsize = (3,3))
    plt.title(f"Rotate image by {i*90} degree")
    plt.imshow(np.rot90(img,i).reshape(28,28))
    plt.show()
# Generate new data
# Create a list with the original train data and the new ones
X_train_gener = []
y_train_gener = []
for i in range(len(X_train)):
    img = X_train[i]
    for r in range(4):
        img = np.rot90(img,r)
        X_train_gener.append(img)
        y_train_gener.append(y_train[i])

# Shuffle the picture and the label on a deterministic way
random.Random(0).shuffle(X_train_gener) 
random.Random(0).shuffle(y_train_gener) 

X_train_gener = np.array(X_train_gener)
y_train_gener = np.array(y_train_gener)

print(f"Total number of training data: {len(X_train_gener)}")
model = LeNet(64, 0.2, X_train=X_train_gener, y_train = y_train_gener)