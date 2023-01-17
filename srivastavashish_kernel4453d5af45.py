##Importing the Requried Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from keras.layers.normalization import BatchNormalization
from tensorflow.keras import optimizers
from keras.models import Sequential
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D,AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
from __future__ import absolute_import, division, print_function, unicode_literals
##Use for finding wheather cpu and gpu are present or not.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
##finding where the datasets are present or not (with respect to the kaggle)
!ls "../input/plant-treat"
##These are the address of the datasets where training,validatation and testing images is present(if it will be run on different platform then the 
## address must be change as accordingly.
base_dir = "../input/plant-treat/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"
test_dir = "../input/plant-treat"
with tf.device("/gpu:0"):
    ##Preprocessing images and doing Data Augmentation.
    train_datagen = ImageDataGenerator(rescale=1./255, ##rescaling the images
                                       shear_range=0.2, ##Shear_range is requried to give the different shape to the image
                                       zoom_range=0.2,  ##zoom the images with some factor in given range
                                       rotation_range = 30, ##rotating images in the range of 0-30 degree
                                       width_shift_range=0.2, ##Shifting the width of the images
                                       height_shift_range=0.2, ##Shifting w.r.t height
                                       fill_mode='nearest') ##Nearest pixel value is chosen
    
    ##For Validatation and testing only preprocessing is required is scaling the images in the range of 0-1
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    ##keeping the batch size to 128 which allow 128 for training at a time and then next 128 images goes for training and so on.
    batch_size = 128
    

    ##loading the training images from the respective directory with size as 224*224 and batch size 128 , and shuffling the images so that the model 
    ##becomes robust and it not just remember the datas.
    training_set = train_datagen.flow_from_directory(base_dir+'/train',
                                                     target_size=(224, 224),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     shuffle=True)

    valid_set = valid_datagen.flow_from_directory(base_dir+'/valid',
                                                target_size=(224, 224),
                                                batch_size=batch_size,
                                                class_mode='categorical')
    testing_set = test_datagen.flow_from_directory(test_dir +'/test',
                                                     target_size=(224, 224))
##Collecting the names of all the 38 classes of the leaf.
class_dict = training_set.class_indices
li = list(class_dict.keys())
i =0
label_image = {}
for j in li:
    label_image[str(i)]=j
    i+=1
print(label_image)
print(li[2])
##declaring a variable with images_size of 224
image_size = 224
##total images used in the training purpose and testing purpose
train_num = training_set.samples
valid_num = valid_set.samples
print(train_num)
print(valid_num)
##Creating a Our_own model Res_VGG for training purpose by using both the build methods ResNet and VGG.

# Initializing the CNN
classifier = Sequential()

##Convolution Step 1and 2
classifier.add(Convolution2D(64, 5, strides = (2, 2), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))
classifier.add(Convolution2D(128, 5, strides = (1, 1), padding = 'valid', activation = 'relu'))
#Average Pooling Step1
classifier.add(AveragePooling2D(pool_size = (2, 2),strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

##Convolution Step 3 and 4
classifier.add(Convolution2D(128, 3, strides = (1, 1), padding='same', activation = 'relu'))
classifier.add(Convolution2D(256, 3, strides = (1, 1), padding='same', activation = 'relu'))
#Max Pooling Step 2
classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

##Convolution Step 5 and 6
classifier.add(Convolution2D(256, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))
#Average Pooling Step 3
classifier.add(AveragePooling2D(pool_size = (2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Convolution Step 7 and 8
classifier.add(Convolution2D(512, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(Convolution2D(512, 3, strides = (1, 1), padding='valid', activation = 'relu'))
#Average Pooling Step 4
classifier.add(AveragePooling2D(pool_size = (2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Convolution Step 9
classifier.add(Convolution2D(512, 3, strides = (1, 1), padding='valid', activation = 'relu'))
#Max Pooling Step 5
classifier.add(MaxPooling2D(pool_size = (2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Flattening Step
classifier.add(Flatten())

# Full Connection Step
classifier.add(Dense(units = 4096, activation = 'relu'))
##adding Dropout of 0.5
classifier.add(Dropout(0.5))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 1000, activation = 'relu'))
##adding Dropout of 0.3
classifier.add(Dropout(0.3))
classifier.add(BatchNormalization())

#Taking 38 neuron as output as we have to predict from only 38 classes of the diseases in the plants.
classifier.add(Dense(units = 38, activation = 'softmax'))
#Getting the summary of the model build above.
classifier.summary()
if(device == 'cuda:0'):
    with tf.device("/gpu:0"):
        #Compiling the Model
        ##Using the Stochastic gradient descent as optimizer with learning rate as 0.005 and decay as 5x10^-5
        classifier.compile(optimizer=optimizers.SGD(lr=0.005, momentum=0.9, decay=5e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        ##fitting images to Res_VGG 
        historymo = classifier.fit_generator(training_set,
                            steps_per_epoch=train_num//batch_size,
                            validation_data=valid_set,
                            epochs=1,
                            validation_steps=valid_num//batch_size)
else :
    #Compiling the Model
        ##Using the Stochastic gradient descent as optimizer with learning rate as 0.005 and decay as 5x10^-5
        classifier.compile(optimizer=optimizers.SGD(lr=0.005, momentum=0.9, decay=5e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        ##fitting images to Res_VGG 
        historymo = classifier.fit_generator(training_set,
                            steps_per_epoch=train_num//batch_size,
                            validation_data=valid_set,
                            epochs=8,
                            validation_steps=valid_num//batch_size)
#saving model in .hdf5 
filepath="Res_VGG_MODEL.hdf5"
classifier.save(filepath)
## To be run only if training is done on the system 
#plotting training values
sns.set()

##Getting training_accuracy datas during the training period for every epochs
acc = historymo.history['accuracy'] 
##Getting validatation_accuracy during the training period for every epochs
val_acc = historymo.history['val_accuracy']
##Training loss for every epochs.
loss = historymo.history['loss']
##Validation loss for every epochs
val_loss = historymo.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
##For saving the plot of accuracy .
plt.savefig('plotaccRes_VGG.png')

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
##For saving the plot of Loss 
plt.savefig('plotlossRes_VGG.png')
plt.show()
## Testing operation
image_path = "../input/plant-treat/test/test/PotatoEarlyBlight5.JPG"
##loading image and resizing it as 224*224*3 as the input training image is of size 224*224*3
new_img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
## Normalizing the image
img = img/255

print("Following is our prediction:")
prediction = model.predict(img)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
d = prediction.flatten()
j = d.max()
#d = sorted(d, reverse = True)
#print(sorted(d, reverse = True))
#print(prediction)
#print(d)
#print(j)
if j>0.75:
    for index,item in enumerate(d):
        #Finding the max accuracy and matching in the confidence shown by the model in predicting the class and after that we will find the index to give
        #it class name
        if item == j:
            class_name = li[index]
            #print(index)
    plt.figure(figsize = (4,4))
    plt.imshow(new_img)
    plt.axis('off')
    plt.title(class_name)
    plt.show()
else:
    print("Accuracy is below 75%")
