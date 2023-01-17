# Importing all the libraries that we will need



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import os

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix

from keras.models import Sequential

from keras.layers import Conv2D , MaxPooling2D , Flatten

from keras.layers import Dense , Dropout , Dense

from keras.preprocessing.image import load_img

from sklearn.utils import shuffle

from random import randint





# We will print all the directory names from our input to get a idea what all we have to make the magic happen



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

# Training data

# filenames_train is here a list to store all our images with their paths

# category_train is here a list to store each image's category



filenames_train = []

category_train = []



# saving the training data path to variable training_data

training_data = os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train")



# As every image is in it's particular directory we will have to enter in each of the respective directory

for dir in training_data:

    

    print(str(dir))

    

    for file in os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train/"+dir):

        

        # Appending the files to filenames_train list

        filenames_train.append("/kaggle/input/intel-image-classification/seg_train/seg_train/"+dir+"/"+file)

        

        # Appending the Categories to category_train

        category_train.append(dir)

        

    print("Process finished for :" + str(dir))

    

    

    

# Creating a Dataframe from our Lists



df_train = pd.DataFrame({

    "File_name":filenames_train,

    "Category":category_train

})
# First five instances 



df_train.head(10)
# Last 10 instances



df_train.tail(10)
# Getting count of each Category



df_train['Category'].value_counts()
# Plotting a bar graph of count for each category



df_train['Category'].value_counts().plot.bar()
# Same process as training

# filenames_test is here a list to store all our images with their paths

# category_test is here a list to store each image's category



filenames_test = []

category_test = []



# saving the testing data path to variable training_data

testing_data = os.listdir("/kaggle/input/intel-image-classification/seg_test/seg_test")



# As every image is in it's particular directory we will have to enter in each of the respective directory

for dir in testing_data:

    

    print(str(dir))

    

    for file in os.listdir("/kaggle/input/intel-image-classification/seg_test/seg_test/"+dir):

        

        # Appending the files to filenames_train list

        filenames_test.append("/kaggle/input/intel-image-classification/seg_test/seg_test/"+dir+"/"+file)

        

        # Appending the Categories to category_train

        category_test.append(dir)

        

    print("Process finished for :" + str(dir))

        

        

# Creating a Dataframe from our Lists    

        

df_test = pd.DataFrame({

    "File_name" : filenames_test,

    "Category" : category_test

})
# First 10 instances of testing data



df_test.head(10)
# Last 10 instances of testing data



df_test.tail(10)
# Shuffling the training dataset



df_train = shuffle(df_train)

df_train.head(10)
# Shuffling the testing dataset



df_test = shuffle(df_test)

df_test.head(10)
# Just to Clarify that images have appropriate labels



i = 0

for index , row in df_train.iterrows():

    if i <=10: 

        print(row['File_name'] + "----->" + row['Category'])

        i += 1
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/mountain/3641.jpg")

plt.imshow(img)

plt.title("Mountain")

plt.show()
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/mountain/3641.jpg")

img = np.array(img)

img.shape
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/sea/2229.jpg")

plt.imshow(img)

plt.title("Sea")

plt.show()
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/sea/2229.jpg")

img = np.array(img)

img.shape
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/glacier/16182.jpg")

plt.imshow(img)

plt.title("Glacier")

plt.show()
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/glacier/16182.jpg")

img = np.array(img)

img.shape
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/buildings/18084.jpg")

plt.imshow(img)

plt.title("Building")

plt.show()
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/buildings/18084.jpg")

img = np.array(img)

img.shape
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/forest/13444.jpg")

plt.imshow(img)

plt.title("Forest")

plt.show()
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/forest/13444.jpg")

img = np.array(img)

img.shape
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/street/13586.jpg")

plt.imshow(img)

plt.title("Street")

plt.show()
img = load_img("/kaggle/input/intel-image-classification/seg_train/seg_train/street/13586.jpg")

img = np.array(img)

img.shape
# X_train and Y_train are lists here to store our training data, where X_train will store the images and Y_train will store the labels



X_train = []

Y_train = []



# The df.iterrows() function will iterate through each row

for index , row in df_train.iterrows():

    

    try:

        

        # Reading the image

        img = cv2.imread(row['File_name'] , cv2.IMREAD_COLOR)

        

        # Resizing the image to our desired dimensions

        img = cv2.resize(img ,(128,128))

    

        # Appending the image as numpy arrays to X_train

        X_train.append(np.array(img))

        

        # Appending the labels to Y_train

        Y_train.append(row['Category'])

    except:

        pass

    

# Just to check how many images were skipped and X_train and Y_train have same data count    

print(len(X_train))

print(len(Y_train))
# X_test and Y_test are lists here to store our testing data, where X_test will store the images and Y_test will store the labels



X_test = []

Y_test = []



# The df.iterrows() function will iterate through each row

for index , row in df_test.iterrows():

    

    try:

        

        # Reading the image

        img = cv2.imread(row['File_name'] , cv2.IMREAD_COLOR)

        

        # Resizing the image to our desired dimensions

        img = cv2.resize(img ,(128,128))

    

        # Appending the image as numpy arrays to X_test

        X_test.append(img)

        

        # Appending the labels to Y_test

        Y_test.append(row['Category'])

    except:

       pass

    

print(len(X_test))

print(len(Y_test))
# We will be plotting some images with the labels



import random



fig , ax = plt.subplots(2,10)

plt.subplots_adjust(bottom=0.3 , top = 0.5 , hspace = 0)

fig.set_size_inches(25,25)



for i in range(0,2):

    for j in range(0,10):

        l = random.randint(0,len(Y_train))

        ax[i,j].imshow(X_train[l])

        ax[i,j].set_title(Y_train[l])

        ax[i,j].set_aspect('equal')
# Converting our datasets to numpy arrays



X_train = np.array(X_train)

Y_train = np.array(Y_train)

X_test = np.array(X_test)

Y_test = np.array(Y_test)
# Reshaping our Y data to be a array of [value , 1]



Y_train = Y_train.reshape(-1,1)

Y_test = Y_test.reshape(-1,1)

ohe = OneHotEncoder()

Y_train = ohe.fit_transform(Y_train)

Y_test = ohe.fit_transform(Y_test)
# Displaying all the Categories 



ohe.categories_
# Getting the output of the transformed data



print(Y_train[1])



# The shape here will be (1 , 6), becuase one hot encoder encoder the data as (1 , numoffeatures array)

# like assuming Building is encoded to [1,0,0,0,0,0]

print(Y_train[1].shape)
model = Sequential()



#First Conv layer

model.add(Conv2D(32 , (3,3) , activation = 'relu' , input_shape = (128 , 128 , 3)))

model.add(MaxPooling2D(pool_size = (2,2)))



#Second Conv layer

model.add(Conv2D(64 , (3,3) , activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



#Third Conv layer

model.add(Conv2D(128 , (3,3) , activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



#Fourth Conv layer

model.add(Conv2D(256, (3,3) , activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



#Fifth Conv layer

model.add(Conv2D(256, (3,3) , activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())



#First Dense layer

model.add(Dense(256 , activation = 'relu'))



# 25% Neurons will get deactivated simulataneously

model.add(Dropout(0.25))



# Second Dense layer

model.add(Dense(64 , activation = 'relu'))



# 50% Neurons will get deactivated simultaneously

model.add(Dropout(0.5))



model.add(Dense(6 , activation = 'softmax'))
# This will give us the summary of our model



model.summary()
# Defining the rules for our model



model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
# Fitting our model to training data



model.fit(X_train , Y_train , epochs = 40 , batch_size = 64)
# Evaluating our model on test data



loss , accuracy = model.evaluate(X_test , Y_test , batch_size = 32)



print('Test accuracy: {:2.2f}%'.format(accuracy*100))
# Transforming the data back to original and saving it to Y_test_labele_data



Y_test_labeled_data  = ohe.inverse_transform(Y_test)
# Assuring the data if converted or not



Y_test_labeled_data[0:5]
# Predicting the values for X_test



Y_pred = model.predict(X_test).round()
# Transforming the data to labels



Y_pred = ohe.inverse_transform(Y_pred)
# Assuring if data is converted or not



Y_pred[0:5]
# Generating a heat map for confusion matrix



import seaborn as sns

from sklearn.metrics import confusion_matrix



x_ticklabels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']



y_ticklabels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']



cm = confusion_matrix(Y_test_labeled_data , Y_pred) 



print(cm)



plt.subplots(figsize = (20,15))



sns.heatmap(cm , xticklabels = x_ticklabels , yticklabels = y_ticklabels)
# Using the data from seg_pred to predict some sample images



seg_pred = os.listdir("/kaggle/input/intel-image-classification/seg_pred/seg_pred")

fig , ax = plt.subplots(5,10)

plt.subplots_adjust(top = 0.7 , bottom = 0.3 , hspace = 0.7)

fig.set_size_inches(25,25)

random_values = []



# Plotting the some random Images

for i in range(5):

    for j in range(10):

        l = random.randint(0, len(seg_pred))

        img = cv2.imread("/kaggle/input/intel-image-classification/seg_pred/seg_pred/"+seg_pred[l])

        ax[i,j].imshow(img)

        ax[i,j].set_title(seg_pred[l])

        ax[i,j].set_aspect('equal')

        random_values.append(seg_pred[l])

    

print(len(random_values))
# Image processing converting the images before feeding them to the model



images_to_be_predicted = []

for i in random_values:

    img = cv2.imread("/kaggle/input/intel-image-classification/seg_pred/seg_pred/"+i)

    img = cv2.resize(img , (128,128))

    images_to_be_predicted.append(np.array(img))

    

print(images_to_be_predicted[1].shape)
images_to_be_predicted = np.array(images_to_be_predicted)
predicted_values = model.predict(images_to_be_predicted).round()
print(predicted_values[0:5])
predicted_values = ohe.inverse_transform(predicted_values)
print(predicted_values[0:5])
# PLotting Image with their predicted labels, you are the witness of how good the model performs



fig , ax = plt.subplots(5,10)

plt.subplots_adjust(top = 0.7 , bottom = 0.3 , hspace = 0.7)

fig.set_size_inches(25,25)

k = 0



for i in range(5):

    for j in range(10):

        ax[i,j].imshow(images_to_be_predicted[k])

        ax[i,j].set_title(predicted_values[k])

        ax[i,j].set_aspect('equal')

        k +=1

        