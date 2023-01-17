import tensorflow as tf
from tensorflow import keras
import glob

import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

import random

import warnings
warnings.filterwarnings('ignore')

# from PIL import Image

class My():
    
    def __init__(self,NUM_EPOCHS):
        self.NUM_EPOCHS = NUM_EPOCHS
        
    
    def getData(self,path,dataset_length):
        files_path = [path+"JMiPOD/*.jpg",path+"JUNIWARD/*.jpg",path+"UERD/*.jpg"]
    
        paths_algorithm = []
        for i in files_path:
            paths_algorithm += glob.glob(i)
        # Getting Random Images from all folders - With Algorithm (JMiPOD,JUNIWARD,UERD) 
        random_pathSelect_Algo =  np.random.randint(0, len(paths_algorithm), dataset_length) 
        with_algorithm_image = []
        for i in random_pathSelect_Algo:
            with_algorithm_image.append(paths_algorithm[i])
        # Getting Random Images from a folder - Without Algorithm (Cover Folder)
        paths_non_algorithm = glob.glob(path+"Cover/*.jpg")
        random_pathSelect_nonAlgo =  np.random.randint(0, len(paths_non_algorithm), dataset_length)
        
        without_algorithm_image = []
        for i in random_pathSelect_nonAlgo:
            without_algorithm_image.append(paths_non_algorithm[i])
            
        # Joining both data with their labels
        train_paths = with_algorithm_image + without_algorithm_image
        train_labels = list([1] * len(with_algorithm_image) + [0] * len(without_algorithm_image))
        # extracting path and converting it to numpy
        images = np.zeros((len(train_paths),128,128,3))
        labels = np.zeros(len(train_labels))

        for i in range(len(train_paths)):
            images[i] = cv2.cvtColor(cv2.resize(cv2.imread(train_paths[i]),(128,128)),cv2.COLOR_BGR2RGB)
            labels[i] = train_labels[i]

        return images,labels
    
    def Inception(self,width, height, depth, classes):
        print("Inception Network ...")
        inputShape = (height, width, depth)
        
        base_model = keras.applications.inception_v3.InceptionV3(weights= None, include_top=False, input_shape= inputShape)
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.7)(x)
        predictions = keras.layers.Dense(classes, activation= 'sigmoid')(x)
        model = keras.models.Model(inputs = base_model.input, outputs = predictions)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
#         model.summary()
        return model
    
    def compile_fit_data_aug(self,model,train_generator,trainX, testX, testY,callbacks_list):

        print("Training network...")
        H = model.fit(X_train,y_train,validation_data=(testX, testY),epochs=self.NUM_EPOCHS,
                    steps_per_epoch=len(trainX)/32, verbose=1,callbacks=callbacks_list)
        print ("Test Data Loss and Accuracy: ", model.evaluate(testX, testY))
        
        return H
    
    def plotImage(self,H):
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.NUM_EPOCHS), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.NUM_EPOCHS), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.NUM_EPOCHS), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.NUM_EPOCHS), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()
        
    def data_augmentation_2(self,trainX,trainY):
        trainDataGenerator = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
        train_generator = trainDataGenerator.flow(trainX,trainY, batch_size=32)
        return train_generator
    
    def checkpoint_model_impovement(self):
        filepath=  "./Model_Save.hdf5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        return callbacks_list
    
Path = "../input/alaska2-image-steganalysis/"
Train_data_generate = 2000
Test_data_generate = 200 
NUM_EPOCHS = 250

my = My(NUM_EPOCHS)
X_train,y_train = my.getData(Path,Train_data_generate)


X_test,y_test = my.getData(Path,Test_data_generate)

model = my.Inception(width=128, height=128, depth=3, classes=1)

# model.fit(X_train,y_train,10
train_generator = my.data_augmentation_2(X_train, y_train)
callbacks_list = my.checkpoint_model_impovement()
                           
H = my.compile_fit_data_aug(model,train_generator,X_train, X_test,y_test,callbacks_list)

my.plotImage(H)

import tensorflow as tf
from tensorflow import keras
import glob

import cv2
import numpy as np
def Inception(width, height, depth, classes):
 
    print("Inception Network ...")
    inputShape = (height, width, depth)

    base_model = keras.applications.inception_v3.InceptionV3(weights= None, include_top=False, input_shape= inputShape)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.7)(x)
    predictions = keras.layers.Dense(classes, activation= 'sigmoid')(x)
    model = keras.models.Model(inputs = base_model.input, outputs = predictions)
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
#         model.summary()
    return model
model = Inception(width=128, height=128, depth=3, classes=1)
model.load_weights('./Model_Save.hdf5')
import pandas as pd
df = pd.read_csv('../input/alaska2-image-steganalysis/sample_submission.csv')
test_path = glob.glob("../input/alaska2-image-steganalysis/Test/*.jpg")
# test_path = test_path[:100]
def data(train_paths,train_labels):
    images = np.zeros((len(train_paths),128,128,3))
    labels = np.zeros(len(train_labels))
    for i in range(len(train_paths)):
        images[i] = cv2.cvtColor(cv2.resize(cv2.imread(train_paths[i]),(128,128)),cv2.COLOR_BGR2RGB)
        labels[i] = train_labels[i]

    return images,labels

test_X, test_y = data(test_path,[0]*len(test_path))
pred = model.predict(test_X, verbose=1)
df['Label'] = pred
df.to_csv('./sample_submission.csv',index=False)




image = cv2.cvtColor(cv2.resize(cv2.imread("../input/alaska2-image-steganalysis/Test/0001.jpg"),(256,256)),cv2.COLOR_BGR2RGB)
image.shape
img.shape
