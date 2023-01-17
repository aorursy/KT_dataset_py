#Import some necessary Modules

import os

import cv2

import keras

import numpy as np

import pandas as pd

import random as rn

from PIL import Image

from tqdm import tqdm

import matplotlib.pyplot as plt

from IPython.display import SVG

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder





from tensorflow.python.keras import backend as K

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.utils import to_categorical

from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from tensorflow.python.keras.utils.vis_utils import model_to_dot

from tensorflow.python.keras.applications.vgg16 import VGG16

from tensorflow.python.keras.applications.resnet50 import ResNet50,preprocess_input

from sklearn.model_selection import train_test_split,KFold, cross_val_score, GridSearchCV

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D,BatchNormalization,Dropout,Conv2D,MaxPool2D



#Resnet-50 has been pre_trained, weights have been saved in below path

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

vgg16_weights_path="../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"



#Display the dir list

print(os.listdir("../input"))
#Transfer 'jpg' images to an array IMG

def Dataset_loader(DIR,RESIZE):

    IMG = []

    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

    for IMAGE_NAME in tqdm(os.listdir(DIR)):

        PATH = os.path.join(DIR,IMAGE_NAME)

        _, ftype = os.path.splitext(PATH)

        if ftype == ".jpg":

            img = read(PATH)

            img = cv2.resize(img, (RESIZE,RESIZE))

            IMG.append(np.array(img)/255.)

    return IMG



benign_train = np.array(Dataset_loader('../input/skin-cancer-malignant-vs-benign/data/train/benign',224))

malign_train = np.array(Dataset_loader('../input/skin-cancer-malignant-vs-benign/data/train/malignant',224))

benign_test = np.array(Dataset_loader('../input/skin-cancer-malignant-vs-benign/data/test/benign',224))

malign_test = np.array(Dataset_loader('../input/skin-cancer-malignant-vs-benign/data/test/malignant',224))
# #Another way to transfer 'jpg' images to an array IMG

# def prep_cnn_data(num, n_x, n_c, path):

#     """

#     This function loads the image jpg data into tensors

#     """

#     # initialize tensors

#     tensors = np.zeros((num, n_x, n_x, n_c))

#     i=0

#     # load image as arrays into tensors

#     for IMAGE_NAME in tqdm(os.listdir(path)):

#         PATH = os.path.join(path,IMAGE_NAME)

#         pic = load_img(PATH)

#         pic_array = img_to_array(pic)

#         tensors[i,:] = pic_array

#         i=i+1

#     # standardize the values by dividing by 255

#     tensors = tensors / 255.

#     return tensors



# benign_train = prep_cnn_data(1440, 224, 3, path='../input/skin-cancer-malignant-vs-benign/data/train/benign')

# malign_train = prep_cnn_data(1197, 224, 3, path='../input/skin-cancer-malignant-vs-benign/data/train/malignant')

# benign_test = prep_cnn_data(360, 224, 3, path='../input/skin-cancer-malignant-vs-benign/data/test/benign')

# malign_test = prep_cnn_data(300, 224, 3, path='../input/skin-cancer-malignant-vs-benign/data/test/malignant')
# Skin Cancer: Malignant vs. Benign

# Create labels

benign_train_label = np.zeros(len(benign_train))

malign_train_label = np.ones(len(malign_train))

benign_test_label = np.zeros(len(benign_test))

malign_test_label = np.ones(len(malign_test))



# Merge data 

X_train = np.concatenate((benign_train, malign_train), axis = 0)

Y_train = np.concatenate((benign_train_label, malign_train_label), axis = 0)

X_test = np.concatenate((benign_test, malign_test), axis = 0)

Y_test = np.concatenate((benign_test_label, malign_test_label), axis = 0)



# Shuffle train data

s = np.arange(X_train.shape[0])

np.random.shuffle(s)

X_train = X_train[s]

Y_train = Y_train[s]



# Split validation data from train data

# x_train, x_val, y_train, y_val = train_test_split(X_train,Y_train,test_size=0.33,random_state=42)

x_train=X_train[1000:]

x_val=X_train[:1000]

y_train=Y_train[1000:]

y_val=Y_train[:1000]



# Shuffle test data

s = np.arange(X_test.shape[0])

np.random.shuffle(s)

X_test = X_test[s]

Y_test = Y_test[s]

# # Display first 15 images of moles, and how they are classified

w=60

h=40

fig=plt.figure(figsize=(15, 15))

columns = 4

rows = 3



for i in range(1, columns*rows +1):

    ax = fig.add_subplot(rows, columns, i)

    if Y_train[i] == 0:

        ax.title.set_text('Benign')

    else:

        ax.title.set_text('Malignant')

    plt.imshow(x_train[i], interpolation='nearest')

plt.show()
# Data auguments

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



# datagen.fit(x_train)
# del model
# Define model with different applications

model = Sequential()

#vgg-16 , 80% accuracy with 100 epochs

# model.add(VGG16(input_shape=(224,224,3),pooling='avg',classes=1000,weights=vgg16_weights_path))

#resnet-50 , 87% accuracy with 100 epochs

model.add(ResNet50(include_top=False,input_tensor=None,input_shape=(224,224,3),pooling='avg',classes=2,weights=resnet_weights_path))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))



model.layers[0].trainable = False

model.summary()
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# Compile model

model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
#Learning rate decay with ReduceLROnPlateau

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.7)
# Train model

batch_size=64

epochs=150

History = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),validation_data=(x_val,y_val),

                              epochs= epochs, steps_per_epoch=x_train.shape[0]//batch_size,verbose=1,callbacks=[red_lr]

                             )
# Testing model on test data to evaluate

lists=[]

y_pred = model.predict(X_test)

for i in range(len(y_pred)):

    if y_pred[i][0]>0.5:

        lists.append(1)

    else:

        lists.append(0)

        

print(accuracy_score(Y_test, lists))
plt.plot(History.history['acc'])

plt.plot(History.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()

i=0

prop_class=[]

mis_class=[]



for i in range(len(Y_test)):

    if(Y_test[i]==lists[i]):

        prop_class.append(i)

    if(len(prop_class)==8):

        break



i=0

for i in range(len(Y_test)):

    if(not Y_test[i]==lists[i]):

        mis_class.append(i)

    if(len(mis_class)==8):

        break
# # Display first 8 images of benign

w=60

h=40

fig=plt.figure(figsize=(18, 10))

columns = 4

rows = 2



def Transfername(namecode):

    if namecode==0:

        return "Benign"

    else:

        return "Malignant"

    

for i in range(len(prop_class)):

    ax = fig.add_subplot(rows, columns, i+1)

    ax.set_title("Predicted result:"+ Transfername(lists[prop_class[i]])

                       +"\n"+"Actual result: "+ Transfername(Y_test[prop_class[i]]))

    plt.imshow(X_test[prop_class[i]], interpolation='nearest')

plt.show()
# # Display first 8 images of benign

w=60

h=40

fig=plt.figure(figsize=(18, 10))

columns = 4

rows = 2

for i in range(len(mis_class)):

    ax = fig.add_subplot(rows, columns, i+1)

    ax.set_title("Predicted result:"+ Transfername(lists[mis_class[i]])

                   +"\n"+"  Actual result: "+ Transfername(Y_test[mis_class[i]]))

    plt.imshow(X_test[mis_class[i]], interpolation='nearest')

plt.show()