import pandas as pd

import tensorflow as tf

import numpy as np

import os





#import tensorflow 

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D, Activation, Flatten, Dropout

from tensorflow.keras.layers import Dense
#check the tensorflow version



tf.__version__
#create a custom class to train the model.



class TrafficSignNet:

    @staticmethod

    def build(width, height, depth, classes):

        model = Sequential()

        inputShape = (height, width, depth)

        chanDim = -1

        

        #layer: Conv -> RELU -> BN -> POOL

        model.add(Conv2D(8, (5,5), padding = "same", input_shape = inputShape))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis = chanDim))

        model.add(MaxPooling2D(pool_size = (2,2)))

        

        # first set of (CONV => RELU => CONV => RELU) * 2 => POOL

        model.add(Conv2D(16, (3,3), padding = "same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Conv2D(16, (3,3), padding = "same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis = chanDim))

        model.add(MaxPooling2D(pool_size = (2,2)))

        

        # second set of (CONV => RELU => CONV => RELU) * 2 => POOL

        model.add(Conv2D(32, (3,3), padding = "same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Conv2D(32, (3,3), padding = "same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Conv2D(32, (3,3), padding = "same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis = chanDim))

        model.add(MaxPooling2D(pool_size = (2,2)))

        

        #FC layers

        model.add(Flatten())

        model.add(Dense(128))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.7))

        

        #FC layers

        model.add(Flatten())

        model.add(Dense(128))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        

        #softmax

        model.add(Dense(classes))

        model.add(Activation("softmax"))

        

        return model
#create a class object



objTSN = TrafficSignNet()
#import necessary packages



import matplotlib.pyplot as plt

plt.style.use("seaborn")



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

from skimage import transform, exposure, io

import random

import os
#define a function to load data from disk



def load_split(basePath, csvPath):

    #intialize the list of data and labels

    data = []

    labels = []

    

    # load the contents of the CSV file, remove the first line (since it contains the CSV header)

    rows = open(csvPath).read().strip().split("\n")[1:]

    random.shuffle(rows)

    

    #loop over the rows of csv file

    for (i, row) in enumerate(rows):

        #check to see if we should show a status update

        if i > 0 and i % 4000 == 0:

            print("[INFO] processed {} total images".format(i))

            

        # split the row into components and then grab the class ID and image path

        (label, imagePath) = row.strip().split(",")[-2:]

        

        # derive the full path to the image file and load it

        imagePath = os.path.sep.join([basePath, imagePath])

        #print(imagePath)

        image = io.imread(imagePath)

        

        #resize the image to be 32x32 pixels, ignoring aspect ratio, and perform CLAHE.

        image = transform.resize(image, (32, 32))

        image = exposure.equalize_adapthist(image, clip_limit = 0.1)

        

        #update the list of data and labels, respectively

        data.append(image)

        labels.append(int(label))

        

    #convert the data and labels into numpy arrays

    data = np.array(data)

    labels = np.array(labels)

    

    #return a tuple of the data and labels

    return (data, labels)
#initialize the hyperparameters

NUM_EPOCHS = 50

INIT_LR = 1e-3

BS = 64



#load the label names

labelNames = ['20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h', '80 km/h end', '100 km/h', '120 km/h', 'No overtaking',

               'No overtaking for tracks', 'Crossroad with secondary way', 'Main road', 'Give way', 'Stop', 'Road up', 'Road up for track', 'Brock',

               'Other dangerous', 'Turn left', 'Turn right', 'Winding road', 'Hollow road', 'Slippery road', 'Narrowing road', 'Roadwork', 'Traffic light',

               'Pedestrian', 'Children', 'Bike', 'Snow', 'Deer', 'End of the limits', 'Only right', 'Only left', 'Only straight', 'Only straight and right', 

               'Only straight and left', 'Take right', 'Take left', 'Circle crossroad', 'End of overtaking limit', 'End of overtaking limit for track']
# derive the path to the training and testing CSV files

trainPath = os.path.sep.join(["../input/gtsrb-german-traffic-sign", "Train.csv"])

testPath = os.path.sep.join(["../input/gtsrb-german-traffic-sign", "Test.csv"])
trainPath
# load the training and testing data

print("[INFO] loading training and testing data...")

(trainX, trainY) = load_split("../input/gtsrb-german-traffic-sign", trainPath)

(testX, testY) = load_split("../input/gtsrb-german-traffic-sign", testPath)
#normalize the images



trainX = trainX.astype("float32")/255.0

testX = testX.astype("float32")/255.0
#one hot encoding of labels



numLabels = len(np.unique(trainY))

trainY = to_categorical(trainY, numLabels)

testY = to_categorical(testY, numLabels)



#take class weight into account

classTotals = trainY.sum(axis = 0)

classWeight = classTotals.max()/classTotals
classWeight
#construct the image data augmentation generator



aug = ImageDataGenerator(

    

    rotation_range = 10,

    zoom_range = 0.15,

    width_shift_range = 0.1,

    height_shift_range = 0.2,

    shear_range = 0.15,

    horizontal_flip = False,

    vertical_flip = False,

    fill_mode = "nearest"

)



#initialize the optimizer and compile the model



opt = Adam(lr = INIT_LR, decay = INIT_LR/(NUM_EPOCHS * 0.5))

model = objTSN.build(width = 32, height = 32, depth = 3, classes=numLabels)

model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
#train the network

H = model.fit_generator(

    aug.flow(trainX, trainY, batch_size = BS),

    validation_data = (testX, testY),

    steps_per_epoch = trainX.shape[0]//BS,

    epochs = NUM_EPOCHS,

    class_weight = classWeight,

    verbose = 1

)
from sklearn.metrics import classification_report
#evaluate network

predictions = model.predict(testX, batch_size = BS)

print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = labelNames))



#save the model

model.save("trainedmodel.h5")
#plot the training loss and accuracy



N = np.arange(0, NUM_EPOCHS)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, H.history["loss"], label = "train_loss")

plt.plot(N, H.history["val_loss"], label = "val_loss")

plt.plot(N, H.history["accuracy"], label = "accuracy")

plt.plot(N, H.history["val_accuracy"], label = "val_acc")

plt.title("Training Loss and Accuracy on DataSet")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc = "lower left")

plt.savefig("plot.png")