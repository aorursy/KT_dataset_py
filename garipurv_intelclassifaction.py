import numpy as np

import pandas as pd

import os

import cv2

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
class_labels = {'buildings':0,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}

class_name = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
batch_size = 128   #do spend time on deciding this parameter so that no data should be lost while training the model

def image_arr(data_dir):

    img_size = (150, 150)

    images = []

    labels = []

    for sub_dir in os.listdir(data_dir):

        print(sub_dir)

        curr_label = class_labels[sub_dir]

        #print(classes)

        for file in os.listdir(data_dir+'/'+sub_dir):

            curr_img = cv2.imread(data_dir+'/'+sub_dir+'/'+file)

            curr_img = cv2.resize(curr_img, img_size)

            images.append(curr_img)

            labels.append(curr_label)

                

    images, labels = shuffle(images, labels)

    images = np.array(images, np.float32)

    labels = np.array(labels, np.int16)

    #data.append((images,labels))

        

    return images, labels
X_test, y_test = image_arr("/kaggle/input/intel-image-classification/seg_test/seg_test")
train_img, train_labels = image_arr("/kaggle/input/intel-image-classification/seg_train/seg_train")
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_img, train_labels, random_state=10, test_size=0.2850)
#Size of Train Data

print("Training Images DataSet Size", X_train.shape)

print("train labels", y_train.shape)



#Validation Size

print("Validation Images DataSet Size", X_valid.shape)

print("Validation labels", y_valid.shape)



#Size of Test Data

print("Test Images DataSet Size", X_test.shape)

print("test labels", y_test.shape)
#Check how spread the data is in train and test set 



train_count = np.bincount(y_train)

valid_count = np.bincount(y_valid)

test_count = np.bincount(y_test)



plt.figure(figsize=(18,12))

ax1 = plt.subplot(221)

ax1.pie(train_count, labels=class_labels.keys(), shadow=True, startangle=90, autopct='%1.1f%%')

ax1.set_title("Train_Data_Count")

ax1 = plt.subplot(222)

ax1.pie(valid_count, labels=class_labels.keys(), shadow=True, startangle=90, autopct='%1.1f%%')

ax1.set_title("Valid_Data_Count")

ax2 = plt.subplot(223)

ax2.pie(test_count, labels=class_labels.keys(), shadow=True, startangle=90, autopct='%1.1f%%')

ax2.set_title("Test_Data_Count")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization

from tensorflow.keras.models import Sequential

import tensorflow as tf
train_image_gen = ImageDataGenerator(rescale = 1./255)

valid_image_gen = ImageDataGenerator(rescale = 1/.255)

test_image_gen = ImageDataGenerator(rescale = 1./255)
train_data_gen = train_image_gen.flow(X_train, y_train, shuffle=False, batch_size=batch_size, seed=100)

valid_data_gen = valid_image_gen.flow(X_valid, y_valid, shuffle=False, batch_size=batch_size, seed=100)

test_data_gen = test_image_gen.flow(X_test, y_test, shuffle=False, batch_size=batch_size, seed=100)
sample_training_images, sample_training_labels = next(train_data_gen)

sample_validation_images, sample_validation_labels = next(test_data_gen)
#View images from Train Data



def plot_images(class_name, img, label):

    fig, axs = plt.subplots(5,5, figsize=(20,20))

    axs = axs.flatten()

    for i, j, ax in zip(img, label, axs):

        ax.imshow(i, cmap=plt.cm.binary)

        ax.set_xlabel(class_name[j])

        ax.set_xticklabels([])

        ax.set_yticklabels([])

    plt.tight_layout()

    plt.show()        
plot_images(class_name, sample_training_images[:30], sample_training_labels[:30])
#Model Building 



model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),

    MaxPooling2D(2,2),

    

    Conv2D(64, (3,3), activation='relu'),

    MaxPooling2D(2,2),

    

    Conv2D(128, (3,3), activation='relu'),

    MaxPooling2D(2,2),

    

    Conv2D(256, (3,3), activation='relu'),

    MaxPooling2D(2,2),

    

    Flatten(),

    Dense(512, activation='relu'),

    Dense(6, activation='softmax'),

])
#Complie the Model 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
#train the model 

#sample size should be such that your training and validation size is completely divisible otherwise model will be seeing 

#few new examples which were left in last epoch and won't be able to train good 

history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=train_data_gen.n//train_data_gen.batch_size,

    epochs=25,

    validation_data=valid_data_gen,

    validation_steps=valid_data_gen.n//valid_data_gen.batch_size,

    verbose=1,

)
#history = model.fit(train_data_gen.x, train_data_gen.y, epochs=30, validation_split=0.30)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(25)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
model.evaluate_generator(valid_data_gen)
model.metrics_names
predictions = model.predict_generator(valid_data_gen.x)

pred_labels = np.argmax(predictions, axis=1)
print(pred_labels[:40])

print(valid_data_gen.y[:40])
#display worngly indetified 25 images using the display method defined above already 

def error_img(class_name, test_images, test_labels, pred_lables):

    BI = (test_labels == pred_lables)

    index = np.where(BI==0)

    print(index[0])

    print("correct ones are", np.where(BI==1)[0].shape)

    print("number of mislabelled images", index[0].shape)

    miss_images = test_images[index]

    miss_labels = pred_labels[index]

    plot_images(class_name, miss_images, miss_labels)
error_img(class_name, valid_data_gen.x, valid_data_gen.y, pred_labels)

#try once without rescaling the data in ImageDataGenerator i.e by 1/.255
#Now check the confusion matrix of classified of batch data



from sklearn.metrics import confusion_matrix

import seaborn as sns



confusion_m = confusion_matrix(valid_data_gen.y, pred_labels)

sns.heatmap(confusion_m, annot=True, xticklabels=class_name, yticklabels=class_name)
train_image_gen_aug = ImageDataGenerator(rescale = 1./255, rotation_range=45, width_shift_range=0.13, 

                                        height_shift_range=0.13, horizontal_flip=True, zoom_range=0.54)

valid_image_gen_aug = ImageDataGenerator(rescale = 1/.255, rotation_range=45, width_shift_range=0.13,

                                         height_shift_range=0.13, horizontal_flip=True, zoom_range=0.54)

test_image_gen_aug = ImageDataGenerator(rescale = 1./255, rotation_range=45, width_shift_range=0.13,

                                        height_shift_range=0.13, horizontal_flip=True, zoom_range=0.54)







train_data_gen_aug = train_image_gen_aug.flow(X_train, y_train, shuffle=False, batch_size=batch_size, seed=100)

valid_data_gen_aug = valid_image_gen_aug.flow(X_valid, y_valid, shuffle=False, batch_size=batch_size, seed=100)

test_data_gen_aug = test_image_gen_aug.flow(X_test, y_test, shuffle=False, batch_size=batch_size, seed=100)
#Model Building 



model_aug = Sequential([

    Conv2D(128, 3,3, input_shape=(150,150,3)),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(3,3),

    Dropout(0.35),

    

    Conv2D(256, (3,3)),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(2,2),

    Dropout(0.25),

    

    Conv2D(256, (3,3)),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(2,2),

    Dropout(0.22),

    

    Conv2D(512, (3,3)),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(5,5),

    Dropout(0.22),

    

    Flatten(),

    Dense(512, activation='relu'),

    Dense(256, activation ='relu'),

    Dense(6, activation='softmax'),

])



#Complie the Model 

model_aug.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_aug.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint



model_path="./cnn_intel.h5"

callbacks = [

    EarlyStopping(monitor='val_accuracy',verbose=1, patience=10, mode='max'),

    ModelCheckpoint(model_path, monitor='val_acc',mode='max', verbose=0, save_best_only=True),

]

#train the model 

#sample size should be such that your training and validation size is completely divisible otherwise model will be seeing 

#few new examples which were left in last epoch and won't be able to train good 

history_aug = model.fit_generator(

    train_data_gen_aug,

    steps_per_epoch=train_data_gen_aug.n//train_data_gen_aug.batch_size,

    epochs=25,

    validation_data=valid_data_gen_aug,

    validation_steps=valid_data_gen_aug.n//valid_data_gen_aug.batch_size,

    verbose=1,

    shuffle=True,

    callbacks=callbacks,

)
acc = history_aug.history['accuracy']

val_acc = history_aug.history['val_accuracy']



loss = history_aug.history['loss']

val_loss = history_aug.history['val_loss']



epochs_range = range(25)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
model.evaluate_generator(valid_data_gen_aug)