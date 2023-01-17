#Defining the folder of the train dataset and creating a list of two categories and using the same categories we define the 

#path to source the images as you can see below.

#Using load_img() function from the keras preprocessing library and setting the color mode to gray scale

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array



import os 

import cv2

import matplotlib.pyplot as plt

train='F:/IMP.DATA/Task/Watermark_train_data'

test='F:/IMP.DATA/Task/Watermark_test_data'



CATEGORIES=['Watermark','No Watermark']

IMG_SIZE=600

for category in CATEGORIES:

    trainpath = os.path.join(train, category)  # directory with our training  pictures

    for img in os.listdir(trainpath):

        trainer = load_img(os.path.join(trainpath,img),color_mode="grayscale")

        train_array= img_to_array(trainer)

        new_train_array = cv2.resize(train_array,(IMG_SIZE,IMG_SIZE))

                        

        

print(new_train_array)
#Defining the folder of the test dataset and creating a list of two categories and using the same categories we define the 

#path to source the images as you can see below.



for category in CATEGORIES:

    testpath = os.path.join(test, category)  # directory with our testing pictures

    for img in os.listdir(testpath):

        tester = load_img(os.path.join(testpath,img),color_mode="grayscale")

        test_array= img_to_array(tester)

        new_test_array = cv2.resize(test_array,(IMG_SIZE,IMG_SIZE))

print(new_test_array)                
#Resizing Images using cv2.resize

IMG_SIZE=600

test_array= img_to_array(tester)

                

new_test_array = cv2.resize(test_array,(IMG_SIZE,IMG_SIZE))

plt.imshow( new_test_array, cmap='gray')

plt.show
#Creating a training data list and creating a function to append the list with new array of resized image arrays with a 

#defined IMG_SIZE.Using img_to_array to convert the images to array and filter out any files which are not images with the

#try and except conditional formatiing

from keras.preprocessing.image import img_to_array

import numpy as np



training_data =[]

IMG_SIZE=600

def create_training_data():

    for category in CATEGORIES:

        trainpath = os.path.join(train,category)

        classnum = CATEGORIES.index(category)

        for img in os.listdir(trainpath):

            try:

                trainer = load_img(os.path.join(trainpath,img),color_mode="grayscale")

                train_array= img_to_array(trainer)

                new_train_array = cv2.resize(train_array,(IMG_SIZE,IMG_SIZE))

                training_data.append([new_train_array,classnum])

            except Exception as e:

                pass

        





create_training_data()



#Figuring out if the images have been properly loaded or not

print(len(training_data))





#Creating a testing data list and creating a function to append the list with new array of resized image arrays with a 

#defined IMG_SIZE.Using img_to_array to convert the images to array and filter out any files which are not images with the

#try and except conditional formatiing

from keras.preprocessing.image import img_to_array

import numpy as np



testing_data =[]

IMG_SIZE=600

def create_testing_data():

    for category in CATEGORIES:

        testpath = os.path.join(test,category)

        classnum = CATEGORIES.index(category)

        for img in os.listdir(testpath):

            try:

                tester = load_img(os.path.join(testpath,img),color_mode="grayscale")

                test_array= img_to_array(tester)

                new_test_array = cv2.resize(test_array,(IMG_SIZE,IMG_SIZE))

                testing_data.append([new_test_array,classnum])

            except Exception as e:

                pass

        





create_testing_data()

#Figuring out if the images have been properly loaded or not



print(len(testing_data))





#Shuffling the assignment of class numbers with the images

import random

random.shuffle(training_data)

for sample in training_data:

    print(sample[1])
#creating a numpy array of the training and testing images and labels list by appending and converting the data type later 



train_images=[]

train_labels=[]

test_images=[]

test_labels=[]



for features , label in training_data:

    train_images.append(features)

    train_labels.append(label)

    

train_images=np.array(train_images).reshape(-1, IMG_SIZE,IMG_SIZE,1)

train_labels=np.array(train_labels)



for features , label in testing_data:

    test_images.append(features)

    test_labels.append(label)

    

test_images=np.array(test_images).reshape(-1, IMG_SIZE,IMG_SIZE,1)

test_labels=np.array(test_labels)



print(len(train_images))



print(len(test_images))
#Using pickle to supply train and test data to use data with other model 

import pickle

pickle_out=open('images.pickle','wb')

pickle.dump(test_images,pickle_out)

pickle_out.close()



pickle_out=open('labels.pickle','wb')

pickle.dump(test_labels,pickle_out)

pickle_out.close()

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

IMG_SIZE=600

model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE ,1)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1, activation='sigmoid')

])

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

history=model.fit(train_images,train_labels, batch_size=2,epochs=5,validation_data=(test_images, test_labels))
prediction=model.predict([test_images])

print((prediction[0][0]))

print((prediction[1][0]))

print((prediction[2][0]))

print((prediction[3][0]))