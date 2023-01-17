# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#inp_dm = 224
# print(X_train)
#X_train = np.zeros((13000,inp_dm,inp_dm,3))
#for i in range(13000):
#    image = cv2.imread('../input/train/train/Img-{}.jpg'.format(i+1))
#    resized_image = cv2.resize(image, (inp_dm, inp_dm)) 
#    X_train[i] =  resized_image
#    if i % 100 == 0:
#        print(i,end=',')

X_train = np.load('../input/trainbeg.npy')

%matplotlib inline
plt.imshow(X_train[10])
#X_test = np.zeros((6000,inp_dm,inp_dm,3))

#for i in range(6000):
#    image = cv2.imread('../input/test/test/Img-{}.jpg'.format(i+1))
#    resized_image = cv2.resize(image, (inp_dm, inp_dm)) 
#    X_test[i] =  resized_image
#    if i % 100 == 0:
#        print(i,end=',')
X_test = np.load('../input/testbeg.npy')
Y_train =  np.load('../input/trainLabels.npy')
print(Y_train.shape)
Y_train = Y_train.reshape(Y_train.shape[0])
np.squeeze(Y_train)
print(Y_train.shape)
print(Y_train)
# from keras.preprocessing.image import ImageDataGenerator
# image_gen = ImageDataGenerator(
#     #featurewise_center=True,
#     #featurewise_std_normalization=True,
#     rescale=1./255,
#     rotation_range=15,
#     width_shift_range=.15,
#     height_shift_range=.15,
#     horizontal_flip=True)

# #training the image preprocessing
# image_gen.fit(X_train, augment=True)

inp_dm = X_train.shape[1]
vgg_model_path = '../input/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
from keras.applications import VGG16
conv_base = VGG16(weights=vgg_model_path,include_top=False,input_shape=(inp_dm,inp_dm, 3))
conv_base.trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True
    if layer.name == 'block4_conv1':
        layer.trainable = True    
    else:
        layer.trainable = False

conv_base.summary()
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Input, BatchNormalization
from keras.layers import Dense

def VGG16_classifier():    
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten()) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
from keras.callbacks import  ReduceLROnPlateau
from keras.utils import to_categorical
model1 = VGG16_classifier()
callbacks_list = [ReduceLROnPlateau(monitor='loss',factor=0.2,patience=3)]

history_vgg = model1.fit(x = X_train/255.,y = to_categorical(Y_train, num_classes=30),batch_size=64,epochs=100,callbacks = callbacks_list, verbose=1)

#model1.fit_generator(image_gen.flow(x = X_train/255.,y = to_categorical(Y_train, num_classes=30),batch_size=32),epochs=200,callbacks = callbacks_list )
from sklearn.metrics import accuracy_score
y_train_predict = np.argmax(model1.predict(x=X_train/255.),axis = 1)

print('\n',y_train_predict)

np.squeeze(Y_train)

print(Y_train)

print("Train accuracy : {}%".format(accuracy_score(Y_train,y_train_predict)))


y_test_predict = model1.predict(x=X_test/255.)

print('\n',y_test_predict)


data_classes = ["antelope","bat","beaver","bobcat","buffalo","chihuahua","chimpanzee","collie","dalmatian","german+shepherd","grizzly+bear",
                "hippopotamus","horse","killer+whale","mole","moose","mouse","otter","ox","persian+cat","raccoon","rat","rhinoceros","seal",
                "siamese+cat","spider+monkey","squirrel","walrus","weasel","wolf"]
label_df = pd.DataFrame(data=y_test_predict, columns= data_classes)

label_df.head(10)

subm = pd.DataFrame()


te_label = pd.read_csv('../input/test.csv')


print(te_label['Image_id'])

subm['image_id'] = te_label['Image_id']

#print(subm.head(10))
subm = pd.concat([subm, label_df], axis=1)

subm.to_csv('submitvgg.csv',index = False)

subm.head(10)