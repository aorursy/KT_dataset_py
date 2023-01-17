# Python program to create 
# Image Classifier using CNN 
# Importing the required libraries 
import cv2
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import math 
import datetime
import time
import random
import gc   #Gabage collector for cleaning deleted data from memory

TRAIN_DIR = '/kaggle/input/training-set/trainData'
TEST_DIR = '/kaggle/input/testingset/testData'
IMG_SIZE = 50
LR = 1e-3

#Getting Train Data - Cats and Dogs
train_dogs = ['/kaggle/input/training-set/trainData/{}'.format(i) 
              for i in os.listdir(TRAIN_DIR) if 'dog' in i]  #get dogs images
train_cats = ['/kaggle/input/training-set/trainData/{}'.format(i) 
              for i in os.listdir(TRAIN_DIR) if 'cat' in i]  #get cat images


#Getting Test Data -
test_imgs = ['/kaggle/input/testingset/testData/{}'.format(i) 
             for i in os.listdir(TEST_DIR)] #get test images


train_imgs = train_dogs[:2000] + train_cats[:2000]  # slice the dataset and use 2000 in each class
random.shuffle(train_imgs)  # shuffle it randomly



#Lets declare our image dimensions
#we are using coloured images. 
nrows = 150
ncolumns = 150
channels = 3  #change to 1 if you want to use grayscale image


#A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images):
    """
    Returns two arrays: 
        X is an array of resized images
        y is an array of labels
    """
    
    X = [] # images
    y = [] # labels
    
    for image in tqdm(list_of_images):
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), 
                            (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image
        #get the labels
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
    
    return X, y
#get the train and label data

X, y = read_and_process_image(train_imgs)

#Lets view some of the pics
plt.figure(figsize=(20,10))
columns = 5
for i in tqdm(range(columns)):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i])
import seaborn as sns
del train_imgs
gc.collect()

#Convert list to numpy array
X = np.array(X)
y = np.array(y)


#Lets plot the label to be sure we just have two class
sns.countplot(y)
plt.title('Labels for Cats and Dogs')
#Lets split the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)
#We can see that our image is a tensor of rank 4, or 
#we could say a 4 dimensional array with dimensions 4000 x 150 x 150 x 3 
#which correspond to the batch size, height, width and channels respectively.

print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)


#clear memory
del X
del y
gc.collect()

#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)
print(ntrain)
print(nval)

#We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout , Activation, Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import applications 
#from tensorflow.keras.utils import to_categorical 


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))  #Dropout for regularization
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  #Sigmoid function at the end because we have just two classes



#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
#Lets create the augmentation configuration
#This helps prevent overfitting, since we are using a small dataset

train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)

#added
train_datagen = ImageDataGenerator(rescale=1./255   #Scale the image between 0 and 1
                                  )


val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale
#Create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

#The training part
#We train for 64 epochs with about 100 steps per epoch

#A total of 3200 images divided by the batch size of 32 will give us 100 steps. 
#This means we going to make a total of 100 gradient update to our model in one pass through the entire training set.

history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=80,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)
#lets plot the train and val curve
#get the details form the history object
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()



plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
#Now lets predict on the first 10 Images of the test set
X_test, y_test = read_and_process_image(test_imgs[0:40]) #Y_test in this case will be empty.
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255) 

out = model.evaluate_generator(val_generator)
print(out)

#need to verify 

Y_pred = model.predict(X_val)
print(Y_pred.shape)
#y_pred = np.argmax(Y_pred, axis=1) - used for multiclass
y_pred = (Y_pred > 0.5) * 1.0
y_pred = y_pred.reshape(y_val.shape)
y_pred.sum()



i = 0
text_labels = []
plt.figure(figsize=(30,20))

for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('dog')
    else:
        text_labels.append('cat')
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + text_labels[i])
    #print(batch[0])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()
from sklearn.metrics import confusion_matrix,classification_report
# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
# Predict the values from the validation dataset



print('Confusion Matrix')
print(confusion_matrix(y_val, y_pred))


print('Classification Report')
target_names = ['Cats', 'Dogs']
print(classification_report(y_val, y_pred,target_names=target_names))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_val, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_val, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_val, y_pred)
print('F1 score: %f' % f1)
 
# kappa
#kappa = cohen_kappa_score(y_val, y_pred)

#print('Cohens kappa: %f' % kappa)
# ROC AUC
#auc = roc_auc_score(y_val, yhat_probs)
#print('ROC AUC: %f' % auc)
# confusion matrix
#matrix = confusion_matrix(y_val, y_pred)
#print(matrix)
