import keras

from keras.models import Sequential,Input,Model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator



import os

import cv2

import numpy as np

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

from IPython.display import SVG
def list_images(basePath, contains=None):

    # return the set of files that are valid

    return list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)



def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):

    # loop over the directory structure

    for (rootDir, dirNames, filenames) in os.walk(basePath):

        # loop over the filenames in the current directory

        for filename in filenames:

            # if the contains string is not none and the filename does not contain

            # the supplied string, then ignore the file

            if contains is not None and filename.find(contains) == -1:

                continue



            # determine the file extension of the current file

            ext = filename[filename.rfind("."):].lower()



            # check to see if the file is an image and should be processed

            if ext.endswith(validExts):

                # construct the path to the image and yield it

                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")

                yield imagePath
def load_images(directory='', size=(150,150)):

    images = []

    labels = []  # Integers corresponding to the categories in alphabetical order

    label = 0

    

    imagePaths = list(list_images(directory))

    

    for path in imagePaths:

        if 'buildings' in path:

            label = 0

            

        elif 'forest' in path:

            label = 1

            

        elif 'glacier' in path:

            label = 2

            

        elif 'mountain' in path:

            label = 3

            

        elif 'sea' in path:

            label = 4

        

        elif 'street' in path:

            label = 5

            

        path = path.replace('\\','')

            

        image = cv2.imread(path) #Reading the image with OpenCV

        image = cv2.resize(image,size) #Resizing the image, in case some are not of the same size

        

        images.append(image)

        labels.append(label)

    

    return shuffle(images,labels,random_state=42) #Shuffles the dataset.
images, labels = load_images(directory='../input/seg_train') #Extract the training images



images = np.array(images)

labels = np.array(labels)
print(images.shape)

print(labels.shape)
label_to_class={

    0 : 'buildings',

    1 : 'forest',

    2 : 'glacier',

    3 : 'mountain',

    4 : 'sea',

    5 : 'street'

}
_,ax = plt.subplots(4,5, figsize = (15,15)) 

for i in range(4):

    for j in range(5):

        ax[i,j].imshow(images[5*i+j])

        ax[i,j].set_title(label_to_class[labels[5*i+j]])

        ax[i,j].axis('off')
train_data_path = '../input/seg_train/seg_train'

test_data_path = '../input/seg_test/seg_test'



size=(150,150)

epochs = 30

batch_size = 32

num_of_train_samples = 14000

num_of_test_samples = 3000



#Image Generator

train_datagen = ImageDataGenerator(rescale=1. / 255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1. / 255)





train_generator = train_datagen.flow_from_directory(train_data_path,

                                                    target_size=size,

                                                    batch_size=batch_size,

                                                    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(test_data_path,

                                                        target_size=size,

                                                        batch_size=batch_size,

                                                        class_mode='categorical',

                                                        shuffle=False)
num_classes = 6



simple_model = Sequential()



simple_model.add(Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))

simple_model.add(MaxPooling2D(2,2))



#The batch normalization allows the model to converge much faster

simple_model.add(BatchNormalization())

simple_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))

simple_model.add(MaxPooling2D(2,2))



simple_model.add(BatchNormalization())

simple_model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))

simple_model.add(MaxPooling2D(5,5))

simple_model.add(Flatten())



simple_model.add(Dense(128, activation='relu'))

simple_model.add(Dense(128, activation='relu'))

simple_model.add(Dropout(rate=0.3))



simple_model.add(Dense(num_classes,activation='softmax'))



simple_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(0.0001),metrics=['accuracy'])



simple_model.summary()
training = simple_model.fit_generator(train_generator,

                                      steps_per_epoch=num_of_train_samples // batch_size,

                                      epochs=epochs,

                                      validation_data=validation_generator,

                                      validation_steps=num_of_test_samples // batch_size)
plt.plot(training.history['acc'])

plt.plot(training.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(training.history['loss'])

plt.plot(training.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
validation_generator = test_datagen.flow_from_directory(test_data_path,

                                                        target_size=size,

                                                        batch_size=batch_size,

                                                        class_mode='categorical',

                                                        shuffle=False)



Y_pred = simple_model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)

y_pred = np.argmax(Y_pred, axis=1)





c=0

for i in range(len(validation_generator.classes)):

  if validation_generator.classes[i]==y_pred[i]:

    c+=1

    

print("Accuracy")

print(c/len(y_pred))



conf_mx=confusion_matrix(validation_generator.classes, y_pred)

print('Confusion Matrix')

print(conf_mx)
def plot_confusion_matrix(matrix):

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(111)

    cax = ax.matshow(matrix)

    fig.colorbar(cax)
plot_confusion_matrix(conf_mx)
row_sums = conf_mx.sum(axis=1, keepdims=True)

norm_conf_mx = conf_mx / row_sums



np.fill_diagonal(norm_conf_mx, 0)



plot_confusion_matrix(norm_conf_mx)
#Shows errors of prediction between two classes, limited to n images

#Not symetric, will show images from class cl1 predicted as images from class cl2

def errors(predictions, cl1, cl2, n):

  _,ax = plt.subplots(n//5,5, figsize = (15,15)) 

  c=0

  for k in range(len(validation_generator.classes)):

    if validation_generator.classes[k]==cl1 and predictions[k]==cl2 and c<n:

      path = validation_generator.filepaths[k]

      image = cv2.imread(path)

      image = cv2.resize(image,size)

      i=c//5

      j=c%5

      ax[i,j].imshow(image)

      ax[i,j].set_title('predicted : '+label_to_class[cl2])

      ax[i,j].axis('off')

      c+=1
#Images of buildings classified as streets

errors(y_pred,0,5,10)
#Images of streets classified as buildings

errors(y_pred,5,0,10)
#Images of glaciers classified as mountains

errors(y_pred,2,3,10)
#Images of mountains classified as glaciers

errors(y_pred,3,2,10)
train_data_path = '../input/seg_train/seg_train'

test_data_path = '../input/seg_test/seg_test'



size=(150,150)

epochs = 30

batch_size = 32

num_of_train_samples = 14000

num_of_test_samples = 3000



#Image Generator

train_datagen = ImageDataGenerator(rescale=1. / 255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1. / 255)





train_generator = train_datagen.flow_from_directory(train_data_path,

                                                    target_size=size,

                                                    batch_size=batch_size,

                                                    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(test_data_path,

                                                        target_size=size,

                                                        batch_size=batch_size,

                                                        class_mode='categorical',

                                                        shuffle=False)
model = Sequential()



model.add(Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))



model.add(BatchNormalization())

model.add(Conv2D(200,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D(5,5))



model.add(BatchNormalization())

model.add(Conv2D(150,kernel_size=(3,3),activation='relu'))



model.add(BatchNormalization())

model.add(Conv2D(150,kernel_size=(3,3),activation='relu'))



model.add(BatchNormalization())

model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))



model.add(BatchNormalization())

model.add(Conv2D(50,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D(5,5))



model.add(Flatten())



model.add(Dense(200,activation='relu'))

model.add(Dense(100,activation='relu'))

model.add(Dense(50,activation='relu'))

model.add(Dropout(rate=0.5))



model.add(Dense(num_classes,activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(0.0001),metrics=['accuracy'])



model.summary()

training_1 = model.fit_generator(train_generator,

                                 steps_per_epoch=num_of_train_samples // batch_size,

                                 epochs=epochs,

                                 validation_data=validation_generator,

                                 validation_steps=num_of_test_samples // batch_size)
plt.plot(training_1.history['acc'])

plt.plot(training_1.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(training_1.history['loss'])

plt.plot(training_1.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
validation_generator = test_datagen.flow_from_directory(test_data_path,

                                                        target_size=size,

                                                        batch_size=batch_size,

                                                        class_mode='categorical',

                                                        shuffle=False)



Y_pred_1 = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)

y_pred_1 = np.argmax(Y_pred_1, axis=1)



c=0

for i in range(len(validation_generator.classes)):

  if validation_generator.classes[i]==y_pred_1[i]:

    c+=1

    

print("Accuracy")

print(c/len(y_pred_1))



conf_mx_1=confusion_matrix(validation_generator.classes, y_pred_1)

print('Confusion Matrix')

print(conf_mx_1)
plot_confusion_matrix(conf_mx_1)
row_sums_1 = conf_mx_1.sum(axis=1, keepdims=True)

norm_conf_mx_1 = conf_mx_1 / row_sums_1



np.fill_diagonal(norm_conf_mx_1, 0)



plot_confusion_matrix(norm_conf_mx_1)
errors(y_pred_1,2,3,10)
errors(y_pred_1,5,0,10)
train_data_path = '../input/seg_train/seg_train'

test_data_path = '../input/seg_test/seg_test'



size=(150,150)

epochs = 20

batch_size = 32

num_of_train_samples = 14000

num_of_test_samples = 3000



#Image Generator

train_datagen = ImageDataGenerator(rescale=1. / 255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1. / 255)





train_generator = train_datagen.flow_from_directory(train_data_path,

                                                    target_size=size,

                                                    batch_size=batch_size,

                                                    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(test_data_path,

                                                        target_size=size,

                                                        batch_size=batch_size,

                                                        class_mode='categorical',

                                                        shuffle=False)
from keras.applications.resnet50 import ResNet50
num_classes = 6



transfer_model = Sequential()



# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is

transfer_model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))



# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation

transfer_model.add(Dense(num_classes, activation = 'softmax'))



# We choose to train the ResNet model

transfer_model.layers[0].trainable = True



transfer_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(0.0001),metrics=['accuracy'])



transfer_model.summary()
#The network has already been trained to recognize patterns, it will converge much faster

#This is why we only train it during 20 epochs

training_2 = transfer_model.fit_generator(train_generator,

                                          steps_per_epoch=num_of_train_samples // batch_size,

                                          epochs=epochs,

                                          validation_data=validation_generator,

                                          validation_steps=num_of_test_samples // batch_size)
plt.plot(training_2.history['acc'])

plt.plot(training_2.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(training_2.history['loss'])

plt.plot(training_2.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
validation_generator = test_datagen.flow_from_directory(test_data_path,

                                                        target_size=size,

                                                        batch_size=batch_size,

                                                        class_mode='categorical',

                                                        shuffle=False)



Y_pred_2 = transfer_model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)

y_pred_2 = np.argmax(Y_pred_2, axis=1)



c=0

for i in range(len(validation_generator.classes)):

  if validation_generator.classes[i]==y_pred_2[i]:

    c+=1

    

print("Accuracy")

print(c/len(y_pred_2))



conf_mx_2=confusion_matrix(validation_generator.classes, y_pred_2)

print('Confusion Matrix')

print(conf_mx_2)
plot_confusion_matrix(conf_mx_2)
row_sums_2 = conf_mx_2.sum(axis=1, keepdims=True)

norm_conf_mx_2 = conf_mx_2 / row_sums_2



np.fill_diagonal(norm_conf_mx_2, 0)



plot_confusion_matrix(norm_conf_mx_2)
errors(y_pred_2,0,5,15)
errors(y_pred_2,2,3,15)