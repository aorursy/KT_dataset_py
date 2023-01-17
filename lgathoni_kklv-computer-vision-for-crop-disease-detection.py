#Data Manipulation Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import re #regular expressions

#Progress bar

from tqdm import tqdm



from datetime import datetime



#Read Images

import os

from skimage import io

from PIL import Image

# import cv2 # When open cv was used, there was an error in getting array from image. Using Pillow eliminated the error.



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns



#Model Pre-processing

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



#Modelling

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from sklearn.metrics import  r2_score,roc_auc_score,f1_score,recall_score,precision_score,classification_report, confusion_matrix,log_loss
# Increase rows and columns visible on the notebook

pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 50)



# import required libraries

import warnings

warnings.filterwarnings("ignore")
#Function to upload the Raw training images

def upload_raw_train_images(image_path, wheat_categories):

    images = []   

    labels = []

    # Loop across the three directories having wheat images.

    for category in wheat_categories:  

        print("Category:",category)

        # Append the wheat category directory into the main path

        full_image_path = image_path +  category + "/"

        # Retrieve the filenames from the all the three wheat directories. OS package used.

        image_file_names = [os.path.join(full_image_path, f) for f in os.listdir(full_image_path)]

        # Read the image pixels

        for file in image_file_names[0:5]:

            image=io.imread(file) #io package from SKimage package

            print(image.shape)

            images.append(np.array(image))

            labels.append(category)

    return images, labels

wheat_categories = ['healthy_wheat', 'stem_rust', 'leaf_rust'] 

raw_train_images, raw_train_labels = upload_raw_train_images('/kaggle/input/cgiar-computer-vision-for-crop-disease/ICLR/train/train/',wheat_categories)
#Function to upload the resized training images

def upload_train_images(image_path, wheat_categories ,height, width):

    images = []

    labels = []

    

    # Loop across the three directories having wheat images.

    for category in wheat_categories:

        

        # Append the wheat category directory into the main path

        full_image_path = image_path +  category + "/"

        # Retrieve the filenames from the all the three wheat directories. OS package used.

        image_file_names = [os.path.join(full_image_path, f) for f in os.listdir(full_image_path)]

        # Read the image pixels

        for file in image_file_names:

            image=io.imread(file) #io package from SKimage package

            # Append image into list

            image_from_array = Image.fromarray(image, 'RGB')

            #Resize image

            size_image = image_from_array.resize((height, width))

            #Append image into list

            images.append(np.array(size_image))

            # Label for each image as per directory

            labels.append(category)

        

    return images, labels



## Invoke the function

#Image resize parameters

height = 256

width = 256

#Get number of classes

wheat_categories = ['healthy_wheat', 'stem_rust', 'leaf_rust'] 

train_images, train_labels = upload_train_images('/kaggle/input/cgiar-computer-vision-for-crop-disease/ICLR/train/train/',wheat_categories,height,width)

#Size and dimension of output image and labels

train_images = np.array(train_images)

train_labels = np.array(train_labels)
print("Shape of training images is " + str(train_images.shape))

print("Shape of training labels is " + str(train_labels.shape))
def show_train_images(images, labels, images_count):

     for i in range(images_count):

        

        index = int(random.random() * len(images))

        plt.axis('off')

        plt.imshow(images[index])

        plt.show()

        

        print("Size of this image is " + str(images[index].shape))

        print("Class of the image is " + str(labels[index]))



#Execute the function

print("Train images, sizes and cass labels")

show_train_images(train_images, train_labels, 3)
# a function to show the image batch

def show_batch_train_images(images,labels):

    plt.figure(figsize=(15,15))

    for n in range(20):

        ax = plt.subplot(5,5,n+1)

        index = int(random.random() * len(images))

        plt.imshow(images[index])

        plt.title(labels[index])

#         plt.title(CLASS_NAMES[labels[n]==1][0].title())

#         print("Size of this image is " + str(images[index].shape))

        plt.axis('off')



show_batch_train_images(train_images,train_labels)
#Categories of Images

pd.Series(train_labels).value_counts().reset_index().values.tolist()
# Plot chart

sns.countplot(train_labels)

plt.show()
#Class Weights

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(train_labels),

                                                 train_labels)



print(np.unique(train_labels))

class_weights
#Label encoding to change 

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

train_labels_enc = label_encoder.fit_transform(train_labels)

train_labels_enc
#Convert the predicted labels to categorical type

train_labels_cat = to_categorical(train_labels_enc)



#Display the categorical training labels

print(train_labels_cat[1])

print(train_labels_cat[300])

print(train_labels_cat[600])
#Normalize the image pixels

train_images = train_images.astype('float32')/255 
# Training to have 90% and validation 10%. High value of training taken so that we have ample training images. 

# The more the images, the better the model

X_train,X_valid,Y_train,Y_valid = train_test_split(train_images,train_labels_cat,test_size = 0.1,random_state=None)



print("X Train count is ",len(X_train),"Shape",X_train.shape, " and Y train count ",len(Y_train), "Shape", Y_train.shape )

print("X validation count is ",len(X_valid), "Shape",X_valid.shape," and Y validation count ", len(Y_valid), "Shape",Y_valid.shape)
#Define the CNN Model

#Sequential API to add one layer at a time starting from the input.

model = Sequential()

# Convolution layer with 32 filters first Conv2D layer.  

# Each filter transforms a part of the image using the kernel filter. The kernel filter matrix is applied on the whole image.

# Relu activation function used to add non linearity to the network.

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))

# Convolution layer with 64 filters second Conv2D layer 

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# Max pooling applied. Reduces the size of the image by half. Is a downsampling filter which looks at the 2 neighboring pixels and picks the maximal value

model.add(MaxPool2D(pool_size=(2, 2)))

# Drop applied as a regularization method, where a proportion of nodes in the layer are randomly ignored by setting their wieghts to zero for each training sample.

# This drops randomly a proportion of the network and forces the network to learn features in a distributed way. This improves generalization and reduces overfitting.

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

# Flatten to convert the final feature maps into a one single 1D vector. Needed so as to make use of fully connected layers after some convolutional/maxpool layers.

# It combines all the found local features of the previous convolutional layers.

model.add(Flatten())

#Dense layer applied to create a fully-connected artificial neural networks classifier.

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

#Neural net outputs distribution of probability of each class.

model.add(Dense(3, activation='softmax'))

model.summary()
#Compilation of the model

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 

                    loss=tf.keras.losses.categorical_crossentropy, 

                    metrics = [tf.keras.metrics.categorical_accuracy])
#Using ten epochs for the training and saving the accuracy for each epoch

history = model.fit(X_train, Y_train, batch_size=32, epochs=12,

                    validation_data=(X_valid, Y_valid),class_weight=class_weights) #,validation_split = 0.2, callbacks=callbacks,
#Display of the accuracy and the loss values

plt.figure(0)

plt.plot(history.history['categorical_accuracy'], label='training accuracy')

plt.plot(history.history['val_categorical_accuracy'], label='val accuracy')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()



plt.figure(1)

plt.plot(history.history['loss'], label='training loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.title('Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
# Create dictionary and dataframe to hold results for various models

dict = {'Model':['Baseline CNN','Mobile Net V2', 'Data Augmentation'], 

        'AUC': [0,0,0],

        'Log Loss':[0,0,0], 

        'F1 score':[0,0,0], 

        'Recall':[0,0,0], 

        'Precision':[0,0,0]} 

df_results = pd.DataFrame(dict,columns = ['Model','Log Loss','AUC','F1 score','Recall','Precision'])





# Function to calculate Results for each model

def model_results(model_type,y_test_data, y_prediction_data, y_test_class, y_pred_class):

    

    index_val = df_results[df_results['Model']==model_type].index

    

    #Asign scores to dataframe

    df_results.loc[index_val,'AUC'] = roc_auc_score(y_test_data, y_prediction_data)

    df_results.loc[index_val,'Log Loss'] = log_loss(Y_valid, y_prediction_data)

    df_results.loc[index_val,'F1 score'] = f1_score(y_test_class, y_pred_class,average='weighted')

    df_results.loc[index_val,'Recall'] = recall_score(y_test_class, y_pred_class,average='weighted')

    df_results.loc[index_val,'Precision'] = precision_score(y_test_class, y_pred_class,average='weighted')



    return(df_results)
#Baseline Prediction

y_prediction = model.predict(X_valid) # make predictions



#Baseline Results

dominant_y_valid=np.argmax(Y_valid, axis=1)

dominant_y_predict=np.argmax(y_prediction, axis=1)



model_results('Baseline CNN',Y_valid, y_prediction,dominant_y_valid,dominant_y_predict)
#Confusion Matrix

import itertools

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=75) 

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            plt.text(j, i, cm[i, j],

            horizontalalignment="center",

            color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



class_names = range(3)

# cm = confusion_matrix(rounded_Y_valid , rounded_Y_predict_trf)

cm = confusion_matrix(dominant_y_valid , dominant_y_predict)

Y_valid, y_prediction

plt.figure(2)

plt.figure(figsize=(5,5))

plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
from keras.layers import SeparableConv2D



#Define the SepConv2D Model

#Sequential API to add one layer at a time starting from the input.

model = Sequential()

# Convolution layer with 32 filters first Conv2D layer.  

# Each filter transforms a part of the image using the kernel filter. The kernel filter matrix is applied on the whole image.

# Relu activation function used to add non linearity to the network.

model.add(SeparableConv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))

# Convolution layer with 64 filters second Conv2D layer 

model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# Max pooling applied. Reduces the size of the image by half. Is a downsampling filter which looks at the 2 neighboring pixels and picks the maximal value

model.add(MaxPool2D(pool_size=(2, 2)))

# Drop applied as a regularization method, where a proportion of nodes in the layer are randomly ignored by setting their wieghts to zero for each training sample.

# This drops randomly a proportion of the network and forces the network to learn features in a distributed way. This improves generalization and reduces overfitting.

model.add(Dropout(rate=0.25))

model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

# Flatten to convert the final feature maps into a one single 1D vector. Needed so as to make use of fully connected layers after some convolutional/maxpool layers.

# It combines all the found local features of the previous convolutional layers.

model.add(Flatten())

#Dense layer applied to create a fully-connected artificial neural networks classifier.

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

#Neural net outputs distribution of probability of each class.

model.add(Dense(3, activation='softmax'))

model.summary()

#Compilation of the model

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 

                    loss=tf.keras.losses.categorical_crossentropy, 

                    metrics = [tf.keras.metrics.categorical_accuracy])



#Using ten epochs for the training and saving the accuracy for each epoch

history_sep = model.fit(X_train, Y_train, batch_size=32, epochs=12,

                    validation_data=(X_valid, Y_valid),class_weight=class_weights)
#Display of the accuracy and the loss values

plt.figure(0)

plt.plot(history_sep.history['categorical_accuracy'], label='training accuracy')

plt.plot(history_sep.history['val_categorical_accuracy'], label='val accuracy')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()



plt.figure(1)

plt.plot(history_sep.history['loss'], label='training loss')

plt.plot(history_sep.history['val_loss'], label='val loss')

plt.title('Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
# Fixed for our Cats & Dogs classes

NUM_CLASSES = 3



# Fixed for Cats & Dogs color images

CHANNELS = 3



IMAGE_RESIZE = 256

RESNET50_POOLING_AVERAGE = 'avg'

DENSE_LAYER_ACTIVATION = 'softmax'

OBJECTIVE_FUNCTION = 'categorical_crossentropy'



# Common accuracy metric for all outputs, but can use different metrics for different output

LOSS_METRICS = ['accuracy']



# EARLY_STOP_PATIENCE must be < NUM_EPOCHS

NUM_EPOCHS = 10

EARLY_STOP_PATIENCE = 3



# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively

# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING

STEPS_PER_EPOCH_TRAINING = 10

STEPS_PER_EPOCH_VALIDATION = 10



# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively

# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input

BATCH_SIZE_TRAINING = 30

BATCH_SIZE_VALIDATION = 30



# Using 1 to easily manage mapping between test_generator & prediction for submission preparation

BATCH_SIZE_TESTING = 1
# from tensorflow.python.keras.applications import ResNet50

from keras_applications.resnet import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense



res_model = ResNet50(weights='imagenet',

                      pooling = RESNET50_POOLING_AVERAGE)
# Create the base model from the pre-trained model MobileNet V2

base_model = tf.keras.applications.MobileNetV2(input_shape=X_train.shape[1:],

                                               include_top=False,

                                               weights='imagenet')
#To use weights in the pre-trained model

base_model.trainable = False 



#Define the pre-trained model

pretrained_model = tf.keras.Sequential([base_model,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(3, activation="softmax")])



pretrained_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.keras.losses.categorical_crossentropy, 

                         metrics = [tf.keras.metrics.categorical_accuracy])



pretrained_model.summary()
#Fit the pretrained model to the  data

history_trf = pretrained_model.fit(X_train, Y_train, epochs=5,batch_size=32 , 

                validation_data=(X_valid, Y_valid), class_weight=class_weights)
#Display of the accuracy and the loss values

plt.figure(0)

plt.plot(history_trf.history['categorical_accuracy'], label='training accuracy')

plt.plot(history_trf.history['val_categorical_accuracy'], label='val accuracy')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()



plt.figure(1)

plt.plot(history_trf.history['loss'], label='training loss')

plt.plot(history_trf.history['val_loss'], label='val loss')

plt.title('Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
#Mobile Net V2 Prediction

y_prediction_trf = pretrained_model.predict(X_valid) # make predictions



#Baseline Results

dominant_y_valid=np.argmax(Y_valid, axis=1)

dominant_y_predict=np.argmax(y_prediction_trf, axis=1)



model_results('Mobile Net V2',Y_valid, y_prediction_trf,dominant_y_valid,dominant_y_predict)
print(classification_report(dominant_y_valid , dominant_y_predict))
#Confusion Matrix

import itertools

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=75) 

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            plt.text(j, i, cm[i, j],

            horizontalalignment="center",

            color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



class_names = range(3)

# cm = confusion_matrix(rounded_Y_valid , rounded_Y_predict_trf)

cm = confusion_matrix(dominant_y_valid , dominant_y_predict)

Y_valid, y_predict_trf

plt.figure(2)

plt.figure(figsize=(5,5))

plot_confusion_matrix(cm, classes=class_names, title='Mobile Net V2 Confusion matrix')
# Generate more image data

from keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(rescale = 1/255, zoom_range = 0.3,horizontal_flip = True,rotation_range = 30)

val_generator = ImageDataGenerator(rescale = 1/255)

train_generator = train_generator.flow(np.array(X_train),Y_train,batch_size = 32,shuffle = False)

val_generator = val_generator.flow(np.array(X_valid),Y_valid,batch_size = 32,shuffle = False)



# Train and test the model

history_idg = pretrained_model.fit_generator(train_generator,

                                   epochs = 10,

                                   shuffle = False, 

                                   steps_per_epoch=3,

                                   validation_steps=1,

                                   validation_data=val_generator,

                                   class_weight=class_weights)
#Display of the accuracy and the loss values

plt.figure(0)

plt.plot(history_idg.history['categorical_accuracy'], label='training accuracy')

# plt.plot(history_idg.history['val_categorical_accuracy'], label='val accuracy')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()



plt.figure(1)

plt.plot(history_idg.history['loss'], label='training loss')

# plt.plot(history_idg.history['val_loss'], label='val loss')

plt.title('Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
# Prediction

y_prediction_idg = pretrained_model.predict(X_valid) # make predictions



logloss = log_loss(Y_valid, y_prediction_idg)

logloss
#Function to upload the test images

def upload_test_images(image_path, height, width):

    test_images = []

    test_image_paths = []

        # Retrieve the filenames from the all the test directory

    test_image_file_names = [os.path.join(image_path, f) for f in os.listdir(image_path)]

        # Read the image pixels

    for file in test_image_file_names:

        test_image=io.imread(file)

        # Append image into list

        test_image_from_array = Image.fromarray(test_image, 'RGB')

        #Resize image

        test_size_image = test_image_from_array.resize((height, width))

        #Append image into list

        test_images.append(np.array(test_size_image))

        test_image_paths.append(file)

    return test_images,test_image_paths



## Invoke the function

#Image resize parameters

height = 256

width = 256

test_images,test_image_paths = upload_test_images('/kaggle/input/cgiar-computer-vision-for-crop-disease/ICLR/test/test/',height,width)

test_images = np.array(test_images)
#Size and dimension of test image

print("Shape of test images is " + str(test_images.shape))

# Check image paths

test_image_paths[0:5]
# use regular expressions to extract the name of image

image_names = []

for i in test_image_paths:

#     name = i

    i = re.sub("[^A-Z0-9]", "", str(i))

    i = i.replace("JPG", "")

    i = i.replace("PNG", "")

    i = i.replace("JPEG", "")

    i = i.replace("JFIF", "")

    i = i.replace("JFIF", "")

    i.strip()

    image_names.append(i)



#View images

image_names[0:5]
#Prediction for all images

y_prediction = model.predict_proba(test_images) # make predictions

y_prediction[400:500]
# Prediction for all images per test image

test_images = np.array(test_images)

preds = []

for img in tqdm(test_images):

    img = img[np.newaxis,:] # add a new dimension

    prediction = pretrained_model.predict_proba(img) # make predictions predict_proba

    preds.append(prediction) 

preds
#healthwheat =0 stem_rust = 2 ,leaf_rst =1

# create a dummy dataset

healthy_wheat = pd.Series(range(610), name="healthy_wheat", dtype=np.float32)

stem_rust = pd.Series(range(610), name="stem_rust", dtype=np.float32)

leaf_rust = pd.Series(range(610), name="leaf_rust", dtype=np.float32)

submission = pd.concat([healthy_wheat,stem_rust,leaf_rust], axis=1)



for i in range(0 ,len(preds)):

    submission.loc[i] = preds[i]
#Append the image names to the result output

submission["ID"] = image_names
submission