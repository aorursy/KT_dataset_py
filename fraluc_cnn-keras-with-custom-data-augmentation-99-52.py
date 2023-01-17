import numpy as np
import cv2
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
def mnistDataPreparation(xy_train):
    """ Takes as input the MNIST train.csv data """
    """ Returns two arrays containing images and labels, properly reshaped for CNN input """
    x_train = xy_train.drop(labels = ["label"],axis = 1)
    x_train = x_train.values.reshape(x_train.shape[0],28,28,1)
    y_train = xy_train["label"]
    y_train = y_train.values.reshape(y_train.shape[0],1)
    return x_train, y_train

def showRandomDigitsAndLabels(x_train, y_train):
    """ Shows twelve random samples from the training set and their labels"""
    fig = plt.figure(figsize = (10,10))
    for i in range(12):
        randomNumber = np.random.randint(len(x_train))
        plt.subplot(3,4,i+1)
        plt.imshow(x_train[randomNumber][:,:,0], cmap='gray')
        plt.title("Label: " + str(y_train[randomNumber][0]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def mnistDataNormalization(x_train,y_train):
    """ Normalizises the MNIST grayscale images and one-hot encodes the labels"""
    x_train = x_train/255.0
    y_train = np_utils.to_categorical(y_train,10)
    return x_train, y_train

class CustomAugmentation(object):
    """ Defines a custom augmentation class"""
    
    kernel = np.ones((3,3),np.uint8)
    
    def __init__(self, erosion = False, dilation = False):
        self.erosion = erosion
        self.dilation = dilation
    
    def __call__(self, img):
        
        randomNumber = np.random.random()
        
        # Erosion and dilation are never applied together
        
        if randomNumber < 0.9:
            pass
        elif randomNumber < 0.95:
            if self.erosion == True:
                # Apply erosion 5% of the time if True
                img = cv2.erode(img,CustomAugmentation.kernel,iterations = 1)
                img = img.reshape(28,28,1)
        else:
            if self.dilation == True:
                # Apply dilation 5% of the time if True
                img = cv2.dilate(img,CustomAugmentation.kernel,iterations = 1)
                img = img.reshape(28,28,1)
                
        return img

def augmentSingleDigit(x_train,datagen):
    """ Applies data augmentation using datagen.flow on a single random sample from x_train to show
        data agumentation effect """
    """ Returns the augmented sample and the corresponding index in the dataset"""
    randomNumber = np.random.randint(len(x_train))
    augmentedDigit = datagen.flow(x_train[randomNumber:randomNumber+1],batch_size = 1)[0][0].reshape(28,28)
    return augmentedDigit, randomNumber

def showAugmentationEffect(x_train, augmentFunction):
    """ Shows the effect of augmentation function augmentFunction on 8 random samples from x_train"""
    plt.figure(figsize = (10,5))
    for i in range(0,8,2):
        plt.subplot(2,4,i+1)
        augmentedDigit, randomNumber = augmentFunction(x_train)
        plt.imshow(x_train[randomNumber].reshape(28,28),cmap = 'gray')
        plt.title(str(randomNumber) + " - Original")
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,4,i+2)
        plt.imshow(augmentedDigit,cmap = 'gray')
        plt.title(str(randomNumber) + " - Augmented")
        plt.xticks([]), plt.yticks([])
    plt.show()
    
def buildModel():
    """ Builds and compiles a new convolutional neural network. Also shows the model summary """
    """ Returns the model and the callbacks array """
    model = Sequential()
    
    # First Convolution Layer
    model.add(Conv2D(32, (5, 5),activation = 'relu', input_shape = (28,28,1)))
    BatchNormalization(axis=1)
    model.add(MaxPooling2D(pool_size=(2,2)))
    BatchNormalization(axis=1)
    
    # 25% Dropout
    model.add(Dropout(0.25))
    
    # Second Convolution Layer
    model.add(Conv2D(64, (5, 5),activation = 'relu', input_shape = (28,28,1)))
    BatchNormalization(axis=1)
    model.add(MaxPooling2D(pool_size=(2,2)))
    BatchNormalization(axis=1)
    
    # 25% Dropout
    model.add(Dropout(0.25))
    
    # Fully connected dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    BatchNormalization()
    model.add(Dense(10, activation='softmax'))
    
    # Shows model summary
    model.summary()
    
    # Compiles the model
    model.compile(optimizer = RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Leraning rate reduction and early stopping that saves best weights
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1,\
                                                 factor=0.5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=2, mode='auto',\
                               restore_best_weights = True)
    callbacks = [early_stopping,learning_rate_reduction]
    
    return model, callbacks

def printModelScore(model, x_val, y_val):
    """ Prints the model score  obtained on validation set (x_val,y_val) """
    """ Returns rounded validation accuracy """
    score = model.evaluate(x_val,y_val,verbose=2)
    metrics = {'loss': 'Validation Loss:', 'acc': 'Validation Accuracy:'}
    print(metrics[model.metrics_names[0]], score[0])
    print(metrics[model.metrics_names[1]], score[1])
    return round(score[1],4)
    
def showWrongPredictions(model,x_val,y_val):
    """ Shows twelve random wrong predictions of model on the set (x_val,y_val) """
    """ Prints predicted label (P) and actual label (A)"""
    
    predictions = model.predict(x_val)
    
    # Initializes array containing array indeces of wrongly predicted samples
    incorrectIndeces = []
    
    # Fills the incorrectIndeces list
    for i in range(len(x_val)):
        prediction = np.argmax(predictions[i])
        actual = np.argmax(y_val[i])
        if prediction != actual:
            incorrectIndeces.append(i)

    print(str(len(incorrectIndeces)) + " out of " + str(len(x_val)) + " test samples classified incorrectly ")
    print("Showing digits with predictions (P) and actual labels (A)...")
    randomOffset = np.random.randint(len(incorrectIndeces)-12)
    
    plt.figure(figsize = (10,10))
    for i, incorrect in enumerate(incorrectIndeces[randomOffset:randomOffset+12]):
        plt.subplot(3,4,i+1)
        plt.imshow(x_val[incorrect].reshape(28,28), cmap='gray', interpolation='none')
        plt.title("P: " + str(np.argmax(predictions[incorrect]))  + " / " + "A: " + str(np.argmax(y_val[incorrect])))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def showPredictionsOnUnlabeled(test, model):
    """ Shows predictions of model on twelve random unlabeled (unseen) samples cointaned in array test"""
    plt.figure(figsize = (10,10))
    randomOffset = np.random.randint(len(test)-12)
    for i in range(12):
        j = randomOffset + i
        intPrediction = np.argmax(model.predict(test[j].reshape(1,28,28,1)))
        plt.subplot(3,4,i+1)
        plt.title("Predicted: " + str(intPrediction))
        plt.imshow(test[j].reshape(28,28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()
# Load train dataset
xy_train = read_csv("../input/train.csv")

# Prepare and reshape images and labels
x_train, y_train = mnistDataPreparation(xy_train)

# Show random images from train sample to check correct preparation
showRandomDigitsAndLabels(x_train, y_train)
# Normalize data
x_train, y_train = mnistDataNormalization(x_train, y_train)

# Split loaded data into training and validation
random_seed = 28
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1,random_state = random_seed)
preprocessor = CustomAugmentation(erosion = True, dilation = True)
datagen = ImageDataGenerator( preprocessing_function = preprocessor, rotation_range = 10, \
                             zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1 )
showAugmentationEffect(x_train, lambda x: augmentSingleDigit(x_train, datagen = datagen))
datagen.fit(x_train)
model, callbacks = buildModel()
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size = 250),
                              epochs = 100, validation_data = (x_val,y_val),
                              verbose = 2, steps_per_epoch = x_train.shape[0] // 250, callbacks = callbacks)
# Prints the model score on the validation set
score = printModelScore(model, x_val, y_val)
# Show twelve random wrong predictions
showWrongPredictions(model,x_val,y_val)
# Show twelve random predictions on the unlabeled data from test.csv
test = read_csv("../input/test.csv")
test = test.values.reshape(test.shape[0],28,28,1)
test = test/255.0
showPredictionsOnUnlabeled(test,model)
import time

model_json = model.to_json()
timecode = time.time()

# Saves the NN structure
with open("mnist_" + str(timecode) + "_" + str(score) + ".json","w") as json_file:
    json_file.write(model_json)

# Saves the model weights
model.save_weights("minst_weights_" + str(timecode) + "_" + str(score) + ".h5")
import pandas as pd
results = model.predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name = "Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("CCN_MNIST_CustomDataAug.csv",index=False)
