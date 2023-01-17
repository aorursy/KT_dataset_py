'''
This script was run on Kaggle platform, so if one want to run this on local machine, he/she will need to install some libraries
and also need to change the directory path accordingly.
'''

'''
We can install these libraries individually or install Anaconda which will automatically install all libraries
To install Anaconda:
1) Download Anaconda install
2) Run 'bash Anaconda-2.x.x-Linux-x86[_64].sh' from command line
'''

'''
In order to increase our model's accuracy, we can increase the number of epochs, increase the data or generate new data, make our 
model more deep, but it would result in increase of time complexity. So, for the sake of simplicity, we will take less numbers and 
values of above parameters for faster run
'''

import numpy as np #sudo pip install numpy scipy
import pandas as pd #sudo pip install pandas
#sudo pip install keras
from keras.models import Sequential 
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from sklearn.utils import shuffle #sudo pip3 install scikit-learn
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
import glob,os
from scipy.misc import imread,imresize
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import zipfile
import shutil
import math
import cv2
from scipy.misc import imread,imresize
import pickle

#This will list all the directories available in current directory
print(os.listdir("../input"))


'''
We will print number of videos of each category
'''

joggingFolder="../input/jogging-20180421t053012z-001/jogging" #path to folder containg jogging videos
walkingFolder="../input/walking-20180421t053059z-001/Walking" #path to folder containing walking videos


count=0
for i in os.listdir(joggingFolder):
    count+=1
    
print("Number of jogging videos: "+str(count))
count = 0

for i in os.listdir(walkingFolder):
    count += 1
    
print ("Number of walking videos: "+str(count))
    
#If there is already a dataset folder we made previously, we will want to remove them in order to start fresh
if "dataset" in os.listdir():
    shutil.rmtree("dataset")
#Convert jogging videos into frames

listing = os.listdir(joggingFolder)
c=1
print ("Converting jogging videos into frames")
for file in listing:
    video = cv2.VideoCapture(joggingFolder+"/"+file)
    #print (video.isOpened())
    framerate = video.get(5)
    if "dataset" not in os.listdir():
        os.makedirs("dataset")
    if "jogging" not in os.listdir("dataset"):
        os.makedirs("dataset/jogging")
        
    count=1
    while video.isOpened():
        frameId = video.get(1)
        success, image = video.read()
        if success!=True:
            break
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        if framerate==0 or frameId % math.floor(framerate) == 0:
            if file[:-4] not in os.listdir("dataset/jogging"):
                os.makedirs("dataset/jogging/"+file[:-4])
            filename = "dataset/jogging/"+file[:-4]+"/image_" + str(count)+".jpg"
            #print (filename)
            cv2.imwrite(filename, image)
            count += 1
    video.release()
    c+=1
print ("Done", c-1)


#Counting total number of frames extracted from jogging videos
su=0
for i in os.listdir("dataset/jogging"):
    for j in os.listdir("dataset/jogging/"+i):
        su+=1
print ("Number of jogging frames: "+str(su))
#Convert walking videos into frames

listing = os.listdir(walkingFolder)
c=1

print ("Converting walking videos into frames")

for file in listing:
    video = cv2.VideoCapture(walkingFolder+"/"+file)
    #print (video.isOpened())
    framerate = video.get(5)
    if "dataset" not in os.listdir():
        os.makedirs("dataset")
    if "walking" not in os.listdir("dataset"):
        os.makedirs("dataset/walking")
        
    count=1
    while video.isOpened():
        frameId = video.get(1)
        success, image = video.read()
        if success!=True:
            break
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        if framerate==0 or frameId % math.floor(framerate) == 0:
            if file[:-4] not in os.listdir("dataset/walking"):
                os.makedirs("dataset/walking/"+file[:-4])
            filename = "dataset/walking/"+file[:-4]+"/image_" + str(count)+".jpg"
            #print (filename)
            cv2.imwrite(filename, image)
            count += 1
    video.release()
    c+=1
print ("Done", c-1)


#Counting total number of frames extracted from jogging videos
su=0
for i in os.listdir("dataset/walking"):
    for j in os.listdir("dataset/walking/"+i):
        su+=1
print ("Number of walking frames: "+str(su))
#Convert extracted frames into numpy arrays

x = []
y = []
count = 0
output = 0

for i in os.listdir("dataset/walking"):
    for j in os.listdir("dataset/walking/"+i):
        image = imread("dataset/walking/"+i+"/"+j)
        x.append(image)
        y.append(0)
for i in os.listdir("dataset/jogging"):
    for j in os.listdir("dataset/jogging/"+i):
        image = imread("dataset/jogging/"+i+"/"+j)
        x.append(image)
        y.append(1)
        
        
x = np.array(x)
y = np.array(y)
print("x",len(x),"y",len(y))
'''
This is the main part where we generate more data using keras.
Also, we will extract features from our images using pretrained ResNet50 model in Keras which was trained on imagenet dataset
Last, we will make our deep neural network by using various CNN and LSTM layers. This model will be trained on the training data 
and also will be continuously checked for accuracy using cross entropy.
'''



batch_size = 32

#We will generate data from out dataset and use scaling for generating more data of similar type
def bring_data_from_directory():
    datagen = ImageDataGenerator(rescale=1. / 255)
    ''
    train_generator = datagen.flow_from_directory(
          'dataset',
          target_size=(224, 224),
          batch_size=batch_size,
          class_mode='categorical',  
          shuffle=True,
          classes=['walking','jogging'])
          
    return train_generator

#We will load pre-trained ResNet50 model for feature extraction
def load_ResNet50_model():
    base_model = ResNet50(weights='../input/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224,224,3))
    print ("Model loaded..!")
    print (base_model.summary())
    return (base_model)

#This function will extract features from out dataset and save it so that we have to not generate is every time we run this script.
#If we have already calculated features once, we can comment all code, except the code code for loading those features
def extract_features_and_store(train_generator, base_model):
    x_generator = np.empty([0,0])
    y_lable = None
    batch = 1
    print (len(train_generator))
    for x,y in train_generator:
        #print (batch)
        if batch == 500:
            break
        if batch%20==0:
            print ("Extracted on batches:",batch)
        batch+=1
        if len(x_generator) == 0:
            x_generator = base_model.predict_on_batch(x)
            y_lable = y
        else:
            x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
            y_lable = np.append(y_lable,y,axis=0)
           # print (y_label)
        #print ("yes")
    x_generator,y_lable = shuffle(x_generator,y_lable)
    print ("Saving features")
    np.save('video_x_ResNet50.npy',x_generator)
    np.save('video_y_ResNet50.npy',y_lable)
    
    print ("Loading features")
    train_data = np.load('video_x_ResNet50.npy')
    train_labels = np.load('video_y_ResNet50.npy')
    train_data,train_labels = shuffle(train_data,train_labels)

    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2], train_data.shape[3])

    return train_data,train_labels

#This is the CNN-LSTM architecture on which we will train our model
def train_model(train_data,train_labels):
    model = Sequential()
    model.add(LSTM(256,dropout=0.2,input_shape=(train_data.shape[1], train_data.shape[2])))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = 25
    model.fit(train_data,train_labels,batch_size=batch_size,epochs=epochs, shuffle=True,verbose=1)
    return model


  
if __name__ == '__main__':
    train_generator = bring_data_from_directory()
    base_model = load_ResNet50_model()
    train_data,train_labels = extract_features_and_store(train_generator, base_model)
    train_model(train_data,train_labels)
    #test_on_whole_videos(train_data,train_labels,validation_data,validation_labels)
    
print ("Done")




