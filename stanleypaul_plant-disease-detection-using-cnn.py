# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # https://numpy.org/



import pandas as pd # https://pandas.pydata.org/ data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

import os  #https://docs.python.org/fr/2.7/library/os.html



# https://www.dataquest.io/blog/sci-kit-learn-tutorial/

# Python libray for machine learning

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import LabelBinarizer



from tensorflow import keras



# https://keras.io/

# Python librabry  specify in deep learning algorithms.

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D  

from keras import backend as K 

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array



# https://matplotlib.org/

# Python library use drawn graphics representation.

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

root = "../input"

pictureList, labelList = [], []

width = 256

height = 256

depth = 3 #Dimension

rgbDim = (3,3)

EPOCHS = 20

BS = 32

METRICS = [

      keras.metrics.TruePositives(name='tp'),

      keras.metrics.FalsePositives(name='fp'),

      keras.metrics.TrueNegatives(name='tn'),

      keras.metrics.FalseNegatives(name='fn'),

      keras.metrics.BinaryAccuracy(name='accuracy'),

      keras.metrics.MeanSquaredError(name='mean_squared_error'),

      keras.metrics.Precision(name='precision'),

      keras.metrics.Recall(name='recall'),

      keras.metrics.AUC(name='auc'),

]





# Image pre-processing function to convert the image into an array of pixel and resize the image. 

def convertImageToArray(imageDir):

    try:

        image = cv2.imread(imageDir)

        if image is not None :

            image = cv2.resize(image, tuple((256,256)))

            return img_to_array(image)

        else:

            return np.array([])

        

    except Exception as e:

        print(f"[ERROR]: {e}")

        return None



#Extract picture inside folder.

try: 

    print("[INFO]: Loading input file ...")

    rootDir = os.listdir(root) # Extract root directiory name.

    print(f"[INFO]: File in root directory are {rootDir} ...")

    

    for plantFolder in rootDir:

        plantFormatFolder = os.listdir(f"{root}/{plantFolder}")

        

        print("[INFO]: Loading Plant Format Folder ...")

        for plantFormat in plantFormatFolder:

            print(f"[INFO]: Current File is {plantFormat} ...")

            plantDiseaseFolder = os.listdir(f"{root}/{plantFolder}/{plantFormat}")

            

            print(f"[INFO]: Current File is {plantDiseaseFolder} ...")

            for plantDisease in plantDiseaseFolder:

                print(f"[INFO]: Load picture from folder {plantDisease} ")

                plantDiseaseArray = os.listdir(f"{root}/{plantFolder}/{plantFormat}/{plantDisease}")

                

                #Take the first 200 picture of each plant diesase folder.

                for picture in plantDiseaseArray[:150]:

                    print(f"[INFO]: Start loading picture {picture} ...")

                    pictureDir = f"{root}/{plantFolder}/{plantFormat}/{plantDisease}/{picture}"

                    if pictureDir.endswith(".jpg") == True or pictureDir.endswith(".JPG") == True:

                        pictureList.append(convertImageToArray(pictureDir))

                        labelList.append(plantDisease)

    print("[INFO]: Picture loading completed.")

    diseaseList = set(labelList)

    print(f"[INFO]: List of disease is {labelList}")

except Exception as e:

    print(f"[ERROR]: {e}")





#Image Generator object is use to increase the number of training image by rotating, 

# shifting, fliping, croping the image, so the image can be re-use (Data augmentation).

# ==> Use to prevent overffiting

print("[INFO]: Image Generator is in production ...")

aug= ImageDataGenerator(rescale = 1./255, 

                        rotation_range=25,

                        width_shift_range = 0.1,

                        height_shift_range = 0.1,

                        shear_range = 0.2, 

                        zoom_range =0.2,

                        horizontal_flip = True,

                        fill_mode= "nearest")

print("[INFO]: Pre-processing Image Generator are completed.")



# Convert binary classification algorithms into multi-class classification.

label_binarizer = LabelBinarizer()

imageLabels = label_binarizer.fit_transform(labelList)



# Split the dataset into a trainingset (70%) and testingset (30%)

print("[INFO]: Splitting dataset into training and testing set")

xTrain, xTest, yTrain, yTest  = train_test_split(pictureList, imageLabels, test_size =0.3, random_state = 42)



#Convert list of training and testing data into array.

xTrain = np.array(xTrain)

xTest = np.array(xTest)



yTrain = np.array(yTrain)

yTest = np.array(yTest)

print("[INFO]: Splitting completed.")





# Created Convolutional Neural Network

print("[INFO]: Creating Convolution Neural Network classifier ...")

classifier = Sequential()



#Founding channel dimension of the input image 

#https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/

if K.image_data_format() == "channels_last":

    inputShape = (height, width, depth)

else :

    inputShape = (depth, width, height)

print("[INFO]: Input image channel dimension detected.")



# Convolutional layers use as an input layer. It will need the input image channel dimension.

# Apply Rectified Linear Unit activation function to the convoltional node.

# https://www.tensorflow.org/tutorials/images/cnn

classifier.add(Conv2D(32, rgbDim, padding="same", activation="relu", input_shape = inputShape))

print("[INFO]: Input layer created.")



classifier.add(MaxPooling2D(pool_size=(2,2))) #(2, 2) will halve the input in both spatial dimension.

classifier.add(Dropout(0.25)) #Prevent overfitting by randomly adding "0".



#Convolutional layer with Rectified Linear Unit activation function

classifier.add(Conv2D(64, rgbDim, activation="relu", padding="same"))

classifier.add(Conv2D(64, rgbDim, activation="relu", padding="same"))





# MaxPooling layer of dimension (2,2)

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))



classifier.add(Conv2D(128, rgbDim, activation="relu", padding="same"))

classifier.add(Conv2D(128, rgbDim, activation="relu", padding="same"))





classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))



classifier.add(Flatten()) # Turn the input matrix (3D) into a vector (1D)

classifier.add(Dense(1024, activation="relu"))# Created an fully connected layers of 1028 nodes.



print("[INFO]: Fully-connected layer created.")



classifier.add(Dropout(0.5))

classifier.add(Dense(len(diseaseList), activation="softmax"))



# Define metrics

classifier.compile(optimizer="adam",

                   loss="binary_crossentropy",

                   metrics=METRICS)



classifier.summary()



print("[INFO]: Start training neural network ...")

history = classifier.fit_generator(

    aug.flow(xTrain, yTrain, batch_size=BS),

    validation_data=(xTest, yTest),

    steps_per_epoch=len(xTrain) // BS,

    epochs=EPOCHS)







acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss =  history.history['loss']

val_loss=  history.history['val_loss']

mean_squared_error = history.history['mean_squared_error']

val_mean_squared_error = history.history['val_mean_squared_error']

auc = history.history['auc']

val_auc = history.history['val_auc']

precision = history.history['precision']

val_precision = history.history['val_precision']

tp = history.history['tp']

fp = history.history['fp']

tn = history.history['tn']

fn = history.history['fn']



sensitivity,specificity, val_sensitivity, val_specificity = [], [], [], []

sensitivity = np.array(tp)/(np.array(tp)+np.array(fn))

specificity = np.array(tn)/(np.array(tn)+np.array(fp))



val_tp = history.history['val_tp']

val_fp = history.history['val_fp']

val_tn = history.history['val_tn']

val_fn = history.history['val_fn']

val_sensitivity = np.array(val_tp)/(np.array(val_tp)+np.array(val_fn))

val_specificity = np.array(val_tn)/(np.array(val_tn)+np.array(val_fp))



epochs = range(1, len(acc) + 1)

#Train and validation accuracy

plt.title('Training and Validation accurarcy')

plt.plot(epochs, acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0, 1])

plt.legend()

plt.figure()



#Train and validation loss

plt.title('Training and Validation loss')

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.ylim([0, 1])

plt.legend()

plt.figure()



#Train and validation Mean Squared Error

plt.title('Training and Validation Mean Squared Error')

plt.plot(epochs, mean_squared_error, 'b', label='Training Mean Squared Error')

plt.plot(epochs, val_mean_squared_error, 'r', label='Validation Mean Squared Error')

plt.xlabel('Epoch')

plt.ylabel('Mean Squared Error')

plt.legend()

plt.figure()



#Train and validation Area Under the Curve

plt.title('Training and Validation AUC')

plt.plot(epochs, auc, 'b', label='Training Area Under the Curve ')

plt.plot(epochs, val_auc, 'r', label='Validation Area Under the Curve')

plt.xlabel('Epoch')

plt.ylabel('AUC')

plt.legend()

plt.figure()



#Train and validation Precision

plt.title('Training and Validation Precision')

plt.plot(epochs, precision, 'b', label='Training Precision')

plt.plot(epochs, val_precision, 'r', label='Validation Precision')

plt.xlabel('Epoch')

plt.ylabel('Precision')

plt.legend()

plt.figure()



#Train and validation Sensitivity

plt.title('Training and Validation Sensitivity')

plt.plot(epochs, sensitivity, 'b', label='Training Sensitivity')

plt.plot(epochs, val_sensitivity, 'r', label='Validation Sensitivity')

plt.xlabel('Epoch')

plt.ylabel('Sensitivity')

plt.legend()

plt.figure()



#Train and validation Specificity

plt.title('Training and Validation Specificity')

plt.plot(epochs, specificity, 'b', label='Training Specificity')

plt.plot(epochs, val_specificity, 'r', label='Validation Specificity')

plt.xlabel('Epoch')

plt.ylabel('Specificity')

plt.legend()

plt.figure()



plt.show()
