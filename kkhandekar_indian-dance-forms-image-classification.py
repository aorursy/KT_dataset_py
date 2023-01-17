#Generic Packages

import numpy as np

import os

import pandas as pd

import random



#SKLearn Library

from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle  

from sklearn.model_selection import train_test_split



#Plotting Libraries

import plotly.express as px

import matplotlib.pyplot as plt



#openCV

import cv2                                 



#Tensor Flow

import tensorflow as tf   

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

from keras.utils import to_categorical

from keras.layers. normalization import BatchNormalization

from keras.optimizers import Adam



#Display Progress

from tqdm import tqdm



#Garbage Collector

import gc
#Define Directory Path

train_images = '../input/indian-dance-classification/Indian_Dance/train/'

test_images = '../input/indian-dance-classification/Indian_Dance/test/'

csv_files = '../input/indian-dance-classification/Indian_Dance/'
# Extracting the Class from the Folder Names in Train Folder



class_names = []

path, dirs, files = next(os.walk(train_images))

class_names = dirs

class_names.sort()
#Class Name Labels 

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)
#Function to Load Images & Labels

def load_data(path):

    

    output = []

    images = []

    labels = []

        

    print("Loading from {}".format(path))

        

    # Iterate through each folder corresponding to a category

    for folder in os.listdir(path):

            label = class_names_label[folder]

            

            

        #Iterate through each image in our folder

            for file in tqdm(os.listdir(os.path.join(path, folder))):

                

                # Get the path name of the image

                img_path = os.path.join(os.path.join(path, folder), file)

                 

                # Open and resize the img

                image = cv2.imread(img_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = cv2.resize(image, IMAGE_SIZE) 

                

                # Append the image and its corresponding label to the output

                images.append(image)

                labels.append(label)

                

    images = np.array(images, dtype = 'float32')

    labels = np.array(labels, dtype = 'int32')   

    output.append((images, labels))

    return output

#Load Training Data

(data_images, data_labels), = load_data(train_images)
#Shuffle The Dataset

data_images, data_labels = shuffle(data_images, data_labels, random_state=25)
#Label Dataset Shape

n = data_labels.shape[0]



print ("Number of examples: {}".format(n))

print ("Each image is of size: {}".format(IMAGE_SIZE))
_, lb_count = np.unique(data_labels, return_counts=True)

data_dance = pd.DataFrame({'Label_Count': lb_count}, index=class_names)

fig = px.bar(data_dance, x=class_names, y='Label_Count', hover_data=['Label_Count'], 

             color_discrete_sequence=px.colors.qualitative.Antique, opacity=0.8, text='Label_Count')



fig.show()

#Data Split - 90% Train & 10% Validation

Image_train, Image_val, Label_train, Label_val = train_test_split(data_images,data_labels, 

                                                                    test_size = 0.1, random_state=42)
Image_train = Image_train / 255.0

Image_val = Image_val / 255.0
#Visualise the data [random image from training dataset]



def display_random_img(class_names, images, labels, val=0):

    index = np.random.randint(images.shape[0])

    plt.figure()

    plt.imshow(images[index])

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    

    if val != 0:

        plt.title(class_names[labels[index]] + ' - {:.4}%'.format(str(val)), fontsize=16)

    else:

        plt.title(class_names[labels[index]], fontsize=16)

        

    plt.show()
#Display Random Image

display_random_img (class_names, Image_train, Label_train)
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(8, activation=tf.nn.softmax)

])
#Summary

model.summary()
#Compile Model

lr = 1e-3 # learn rate

model.compile(optimizer = Adam(lr), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
#Train

history = model.fit(Image_train, Label_train, batch_size=10, epochs=10, validation_split = 0.1)
acc = model.evaluate(Image_val,Label_val, verbose=1)
predictions = model.predict(Image_val) 

pred_labels = np.argmax(predictions, axis = 1) 

conf = 100*np.max(predictions)



display_random_img(class_names, Image_val, pred_labels, conf)
#Function to Load Test Data (Unseen)

def load_Testdata(path):

    

    images = []

      

    print("Loading from {}".format(path))

    

    for file in tqdm(os.listdir(path)):

        # Get Image Path    

        img_path = os.path.join(path, file)

        

        # Open and resize the img

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, IMAGE_SIZE) 

                

        # Append the image and its corresponding label to the output

        images.append(image)

                

    images = np.array(images, dtype = 'float32')

    return images

#Load Images from Test Folder

(Test_Images) = load_Testdata(test_images)
#Scale Data

Test_Images = Test_Images / 255.0
#Making Predictions

Test_Pred = model.predict(Test_Images)          # Vector of probabilities

Test_Pred_lb = np.argmax(Test_Pred, axis = 1) # We take the highest probability

Test_conf = 100*np.max(Test_Pred)

display_random_img(class_names, Test_Images, Test_Pred_lb,Test_conf)