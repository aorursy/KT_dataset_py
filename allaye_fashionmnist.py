import keras



from keras.models import Sequential, load_model

from keras.layers import Dense,Convolution2D, Flatten, MaxPooling2D, Dropout

from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sbn

import cv2 as cv

import random

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')
#loading the dataset

training_set = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

testing_set = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
training_set.head()
testing_set.tail()
training = np.array(training_set, dtype = 'float32')

testing = np.array(testing_set, dtype = 'float32')
i = random.randint(0,6001)
plt.imshow(training[i,1:].reshape(28,28))
fig , axe = plt.subplots(15,15, figsize=(17,17)) #create a plot of 15 row and 15 column each with a size of 17x17

axe = axe.ravel() #reareange the array of axe as 1D

len_train = len(training_set) #getting the length of the training set



for i in np.arange(0, 15*15): #creating a loop the runs for 225 time, which is the amonth of subplot we created

    

    index = np.random.randint(0,len_train) #creating a random numbers from 0 to lenght og training set, we will use to index the image

    

    axe[i].imshow(training[index, 1:].reshape(28,28))

    axe[i].set_title(training[index,0], fontsize = 8)

    axe[i].axis('off')

plt.subplots_adjust(hspace=0.4)
x_train = training[:,1 :]/255  # picking all row from the second column to the last and normalizing the values by dividing by 255

y_train = training[:,0] # picking just the label column from the training set
x_test = testing[:,1 :]/255

y_test = testing[:, 0]
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size= 0.2, random_state = 12345)

#resize the image 

x_train = x_train.reshape(x_train.shape[0], 28,28,1)   

x_validate = x_validate.reshape(x_validate.shape[0], 28,28,1) 

x_test = x_test.reshape(x_test.shape[0], *(28,28,1)) # since the reshape expects just normal int but we are passing tuple so we use asterick 
a = x_train.shape

a
#creating the CNN layers 

classifier = Sequential()

classifier.add(Convolution2D(32, (3,3), input_shape=(28,28,1), activation='relu')) # add a 32 feature detector to the CNN, with a 3x3 shape

classifier.add(MaxPooling2D(pool_size=(2,2))) #appling maxpooling od size 2x2 to the CNN



classifier.add(Flatten()) # we flatten the CNN into one single vector
#connect the CNN to the ANN

classifier.add(Dense(output_dim=32, activation='relu')) # adding the hidden layer of 32 neurons

classifier.add(Dense(output_dim=32, activation='relu')) # adding the hidden layer of 32 neurons

classifier.add(Dense(output_dim=32, activation='relu')) # adding the hidden layer of 32 neurons

classifier.add(Dense(output_dim=32, activation='relu')) # adding the hidden layer of 32 neurons

classifier.add(Dense(output_dim=32, activation='relu')) # adding the hidden layer of 32 neurons

classifier.add(Dense(output_dim=32, activation='relu')) # adding the hidden layer of 32 neurons

classifier.add(Dense(output_dim=32, activation='relu')) # adding the hidden layer of 32 neurons

classifier.add(Dense(output_dim=32, activation='relu')) # adding the hidden layer of 32 neurons

classifier.add(Dense(output_dim=10, activation='sigmoid')) #adding output layer of 10 neurons 
classifier.compile(loss = 'sparse_categorical_crossentropy', optimizer= Adam(lr=0.001), metrics= ['sparse_categorical_accuracy'])
classifier.fit(x_train,

               y_train,

               batch_size=512,

               epochs=50,

               verbose=1,

               validation_data=(x_validate,y_validate)

              )

classifier.save('fashion_class.h5')
evalu = classifier.evaluate(x_test,y_test) 

print('Test accuracu is {:0.3f}'.format(evalu[1]))
predicted_class = classifier.predict_classes(x_test) #getting all the predicted class of the x_test
fig,axe = plt.subplots(5,5, figsize=(12,12))# creating a 5x5 graph of size 12x12 each

axe = axe.ravel()#flatten the value of axe into a vector format



for i in np.arange(0,5*5):# we will loop through 25 times

    axe[i].imshow(x_test[i].reshape(28,28)) #for each index of graph we add a image from x_test and reshape the image

    axe[i].set_title('prediction class = {:0.1f}\n True class = {:0.1f}'.format(predicted_class[i], y_test[i]))

    #showing the predicted class and the true value of the predicted class

    axe[i].axis('off')

plt.subplots_adjust(wspace =0.55)
confmetrix = confusion_matrix(y_test, predicted_class)

plt.figure(figsize=(14,10))

sbn.heatmap(confmetrix, annot=True)
class_names = ['class {}'.format(i)for i in range(0,10)]

print(classification_report(y_test,predicted_class, target_names= class_names))