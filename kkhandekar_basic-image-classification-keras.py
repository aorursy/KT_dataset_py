#Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,re

import random

from PIL import Image 



# TensorFlow / Keras Libraries

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

from keras.utils import to_categorical

from keras.layers. normalization import BatchNormalization



#Plotting Libraries

import matplotlib.pyplot as plt



#SKLearn Libraries

from sklearn.model_selection import train_test_split



#Garbage Collect

import gc



#Show Progress Library

from tqdm import tqdm
#Define Directory Path

train_images = '../input/baldclassificationselected/data/input/BaldClassification/'

test_images = '../input/baldclassificationselected/data/input/BaldClassification/test/'

csv_files = '../input/baldclassificationselected/data/input/BaldClassification/'

#Loading Training Data

trainData_url = f'{csv_files}/train.csv'

train_data = pd.read_csv(trainData_url, header='infer')
#Creating Test-Data (image names) from test-folder

test_d = []

for subdir, dirs, files in os.walk(test_images):

    for f in files:

        test_d.append(f)



test_data = pd.DataFrame(test_d, columns= ["TestData"])
#Check for records

print("Total Records in Training Dataset: ", train_data.shape[0])

print("Total Records in Testing Dataset: ", test_data.shape[0])
#Check for null values

print("Null/Missing Values in Training Dataset: ",train_data.isna().sum())
#Check for total labels in Training Dataset

train_data.groupby('label').size()
#Load Train Images 

train_image = []

for i in tqdm(range(train_data.shape[0])):

    img = image.load_img(f'{train_images}' + train_data['image_path'][i],target_size=(150,150,3))

    img = image.img_to_array(img)

    #img = img/255 

    train_image.append(img)

    

#Array of Training Images    

training_images = np.array(train_image)
#Load Test Images

test_image = []

for i in tqdm(range(test_data.shape[0])):

    img = image.load_img(f'{test_images}' + test_data['TestData'][i],target_size=(150,150,3))

    img = image.img_to_array(img)

   # img = img/255 

    test_image.append(img)

    

#Array of Training Images    

testing_images = np.array(test_image)
training_images = training_images / 255.0



testing_images = testing_images / 255.0
#Visualizing a random image



def random_img():

    

    fig = plt.figure(figsize=(10,10))

    plt.subplots_adjust(hspace = 0.9)

    

    plt.subplot(221)

    ax1 = plt.imshow(training_images[random.randint(0, 6720)])

    plt.colorbar()

    plt.title("Random Image from Training set", fontsize=12)

    plt.grid(False)



    plt.subplot(222)

    ax2 = plt.imshow(testing_images[random.randint(0, 2729)])

    plt.colorbar()

    plt.title("Random Image from Testing set", fontsize=12)

    plt.grid(False)

    

    

    plt.show()
#Visualize a random image from training image array

random_img()
training_trgt = np.array(train_data.drop('image_path',axis=1))
print("Total Records in Training Target Dataset: ", training_trgt.shape[0])
model = Sequential()



model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())



model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())



model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())



model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())



model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())



model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(2, activation = 'softmax'))
#model = keras.models.load_model("../input/output/ImgClassificationModel")
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='nadam', metrics = ['accuracy'])
#Training the model on training data i.e. Training Images & Training Labels aka Target

model.fit(training_images, training_trgt, batch_size = 50, epochs = 10, verbose = 1)
#Predicting Labels

predictions_class = model.predict_classes(testing_images)

#Predicting Values

predictions_val = model.predict(testing_images)



#Storing the Predicted Labels & Values to Test Dataset

test_data['Predictions'] = predictions_class

test_data['PredVals'] = predictions_val.tolist()

def display_random_prediction():

    

    class_names = ['Bald','Not Bald']  # Define the Class Names for Binary Classification 

    

    index = np.random.randint(test_data.shape[0])   #Randomly generating an index number

    

   

    fn = test_data.TestData[index]      #Storing the filename for randomly generated index

    lb = test_data.Predictions[index]   #Storing the Predictions for randomly generated index





    

    #Load Image

    test_img = image.load_img(f'{test_images}' + fn,target_size=(150,150,3))

    test_img = image.img_to_array(test_img)

    test_img = test_img/255.0

    

    #Plot Image

    plt.figure()

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(test_img)

    

    # Add the image to a batch where it's the only member.

    test_img = (np.expand_dims(test_img,0))



    # Make Prediction on the singl image

    pred_single_val = model.predict(test_img)

    pred_single_lb = np.argmax(pred_single_val[0])

    

    #print("value: ",pred_single_val, "----", "class:",pred_single_class)

    



    #Plot Title with Prediction

    plt.title("{} - {:2.1f}%".format(class_names[pred_single_lb], 100*np.max(pred_single_val) ), fontsize=13)

    

    
display_random_prediction()