# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import backend

import tensorflow as  tf



# Model architecture

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D

from keras.layers import MaxPool2D, Activation, MaxPooling2D

from keras.layers.normalization import BatchNormalization



# Annealer

from keras.callbacks import LearningRateScheduler



# Data processing

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical

from keras.preprocessing import image



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns





# Progressor

from tqdm import tqdm

import h5py

# Read File

sample_submission = pd.read_csv('../input/cifar-10/sampleSubmission.csv')

train_labels = pd.read_csv('../input/cifar-10/trainLabels.csv')



print("Number of training sample: ",train_labels.shape[0])

print("Number of test samples: ", sample_submission.shape[0])
import h5py

with h5py.File('../input/cifardata/train_data.h5', 'r') as file:

    #for key in file.keys():

     #   print(key)

    train_ids = pd.DataFrame(np.array(np.squeeze(file['train_ids'])),columns=['id'])

    train_data = np.array(file['train_images']).reshape(-1, 32, 32, 3)

    

with h5py.File('../input/cifardata/test_data_1.h5', 'r') as file:

    test_ids_1 = pd.DataFrame(np.array(np.squeeze(file['test_ids'])),columns=['id'])

    test_data_1 = np.array(file['test_images']).reshape(-1, 32, 32, 3)

    

with h5py.File('../input/cifardata/test_data_2.h5', 'r') as file:

    test_ids_2 = pd.DataFrame(np.array(np.squeeze(file['test_ids'])),columns=['id'])

    test_data_2 = np.array(file['test_images']).reshape(-1, 32, 32, 3)

    

    

with h5py.File('../input/cifardata/test_data_3.h5', 'r') as file:

    test_ids_3 = pd.DataFrame(np.array(np.squeeze(file['test_ids'])),columns=['id'])

    test_data_3 = np.array(file['test_images']).reshape(-1, 32, 32, 3)

    

    

with h5py.File('../input/cifardata/test_data_4.h5', 'r') as file:

    test_ids_4 = pd.DataFrame(np.array(np.squeeze(file['test_ids'])),columns=['id'])

    test_data_4 = np.array(file['test_images']).reshape(-1, 32, 32, 3)

    

    

with h5py.File('../input/cifardata/test_data_5.h5', 'r') as file:

    test_ids_5 = pd.DataFrame(np.array(np.squeeze(file['test_ids'])),columns=['id'])

    test_data_5 = np.array(file['test_images']).reshape(-1, 32, 32, 3)

    

    

with h5py.File('../input/cifardata/test_data_6.h5', 'r') as file:

    test_ids_6 = pd.DataFrame(np.array(np.squeeze(file['test_ids'])),columns=['id'])

    test_data_6 = np.array(file['test_images']).reshape(-1, 32, 32, 3)

    

test_data = np.concatenate([test_data_1, test_data_2, test_data_3, test_data_4, test_data_5, test_data_6], axis=0)

test_ids = pd.concat([test_ids_1, test_ids_2, test_ids_3, test_ids_4, test_ids_5, test_ids_6], axis=0).reset_index()

#print(test_data.head())

print(test_ids.head())
# check load test data is consistent are not

test_ids.id.value_counts().sort_index() - sample_submission.id.value_counts().sort_index()
sample_submission.id.value_counts().sort_index()
# check train data is consistent or not

sum(train_ids.id == train_labels.id)
# shape of data

print(train_data.shape)

print(test_data.shape)

print(train_ids.shape)

print(test_ids.shape)
# Distribution of classes in training samples



train_labels.label.value_counts().plot(kind='bar', title='Distribution of classes')
# Function to reshape and scaling image

def Scale_Reshape(x):

    x_min = x.min(axis=(1, 2), keepdims=True)

    x_max = x.max(axis=(1, 2), keepdims=True)



    x = (x - x_min)/(x_max-x_min)

    

    x = x.reshape(-1, 32, 32, 3)

    return x

# Training data processing

train = Scale_Reshape(train_data)



# Test data processing

test = Scale_Reshape(test_data)



# Label processing



Y=train_labels['label']

# convert to one-hot

Y = pd.get_dummies(Y)





print("train shape: ", train.shape)

print("test shape: ", test.shape)

print("one-hot label shape: ", Y.shape)
# Label encoding

from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

label_int = train_labels[['label']].copy()

label_int.label = lb_make.fit_transform(label_int.label)

# visualizing training samples

plt.figure(figsize=(15,5))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(train_data[i].reshape((32, 32, 3)),cmap=plt.cm.hsv)

    plt.title(f"{train_labels.label[i]}")

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show()
# visualizing test samples

plt.figure(figsize=(15,5))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(test[i].reshape((32,32, 3)),cmap=plt.cm.hsv)

    plt.axis('off')

plt.subplots_adjust(wspace=0.0, hspace=0.0)

plt.show()
# split training and validation set.

X_train, X_val, Y_train, Y_val = train_test_split(train, Y, random_state=0, test_size=0.1)

print("X_train shape: ", X_train.shape)

print("Y_train shape: ", Y_train.shape)

print("X_val shape: ", X_val.shape)

print("Y_val shape: ", Y_val.shape)
# BUILD CONVOLUTIONAL NEURAL NETWORKS



model = Sequential()



model.add(Conv2D(32,  kernel_size = 3,kernel_initializer='he_normal', activation='relu', input_shape = (32, 32, 3)))

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size = 3, kernel_initializer='he_normal', strides=1, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, kernel_size = 3, strides=1, kernel_initializer='he_normal' ,padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, kernel_size = 3,kernel_initializer='he_normal', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))





model.add(Flatten())

model.add(Dense(512,kernel_initializer='he_normal', activation = "relu"))

model.add(Dropout(0.2))

model.add(Dense(10, kernel_initializer='glorot_uniform', activation = "softmax"))





# Compile the model

model.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=["accuracy"])



# Summary of model

model.summary()

"""

import math

# learning rate schedule

def step_decay(epoch):

    initial_lrate = 0.1

    drop = 0.5

    epochs_drop = 3.0

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate



# learning schedule callback

annealer = LearningRateScheduler(step_decay)

callbacks_list = [annealer]

"""
# data augumetation

datagen = ImageDataGenerator(

        rotation_range=0,  

        zoom_range = 0.0,  

        width_shift_range=0.1, 

        height_shift_range=0.1,

        horizontal_flip=False)



# data generator model to train and validation set

batch_size_1 = 500

train_gen = datagen.flow(X_train, Y_train, batch_size=batch_size_1)

val_gen = datagen.flow(X_val, Y_val, batch_size=batch_size_1)
# visualizing augumented image

X_train_augmented = X_train[9,].reshape((1,32,32,3))

Y_train_augmented = np.array(Y_train.iloc[9,:]).reshape((1,10))

plt.figure(figsize=(15,4.5))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    X_train2, Y_train2 = datagen.flow(X_train_augmented,Y_train_augmented).next()

    plt.imshow(X_train2[0].reshape((32,32,3)),cmap=plt.cm.gray)

    plt.axis('off')

    if i==9: X_train_augmented = X_train[2000,].reshape((1,32,32,3))

    if i==19: X_train_augmented = X_train[1180,].reshape((1,32,32,3))

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
# traing parameters

epochs = 2

batch_size = 32
# Fit the model

history = model.fit_generator(train_gen, 

                              epochs = epochs, 

                              steps_per_epoch = X_train.shape[0] // batch_size,

                              validation_data = val_gen,

                              validation_steps = X_val.shape[0] // batch_size,

                              

                              verbose=1)
final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=1)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1, figsize=(15, 5))

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# making predictions

prediction = model.predict_classes(X_val)



# PREVIEW PREDICTIONS

plt.figure(figsize=(20,8))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X_val[i].reshape((32,32,3)),cmap=plt.cm.gray)

    plt.title(f"predict={Y.columns.values[prediction[i]]}")

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show()
# prediction on test

prediction = model.predict_classes(test)

#test_ids.head()

test_ids['label'] =str(0)

print(test_ids.head())

for i in tqdm(range(sample_submission.shape[0])):

    test_ids.loc[i, 'label'] = Y.columns.values[prediction[i]]



print(test_ids.head())

    
final_file = pd.merge((sample_submission, teat_ids), how='inner', left_on='id', right_on='id')

final_file.to_csv('samplefile.csv')
# make submission

"""

for i in tqdm(range(sample_submission.shape[0])):

    if i<100000 and i>=0:

        sample_submission.loc[i, 'label'] = Y.columns.values[prediction_1[i]]

    elif i <200000 and i>=100000:

        sample_submission.loc[i, 'label'] = Y.columns.values[prediction_2[i-100000]]

    elif i<300000 and i >=200000:

        sample_submission.loc[i, 'label'] = Y.columns.values[prediction_3[i-200000]]

"""       

#sample_submission.to_csv('sampleSubmission.csv')

#sample_submissions.head(20)
# distribution of predicted class

#sample_submission.label.value_counts().plot(kind='bar', title='Pridicted class ditribution', figsize=(15, 4.5))
"""

# create model

model=Sequential()



#model.add(Lambda(standardize,input_shape=(28,28,1)))    

model.add(Conv2D(filters=32, kernel_size = (5,5), activation="relu", input_shape=(32,32,1)))

model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

#model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())    

model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))



model.add(MaxPooling2D(pool_size=(2,2)))

    

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.25)) 

model.add(Dense(10,activation="softmax"))

"""