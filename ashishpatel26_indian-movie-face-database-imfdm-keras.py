# Import libraries# Impor 
import os,cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import keras

from keras.utils import np_utils

from keras import backend as K

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Reshape, Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, UpSampling2D
#Define Datapath
data_path = '../input/train/'
data_dir_list = os.listdir(data_path)

# img_rows=256
# img_cols=256
num_channel=1

num_epoch=10

img_data_list=[]


for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(32,32))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

train_x = np.stack(img_data_list)
train_x.shape

#Define Datapath
data_path1 = '../input/test/'
data_dir_list = os.listdir(data_path1)

# img_rows=256
# img_cols=256
num_channel=1

num_epoch=10

img_data_list1=[]


for dataset in data_dir_list:
    img_list=os.listdir(data_path1+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path1 + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(32,32))
        img_data_list1.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

test_x = np.stack(img_data_list1)
test_x.shape
train = pd.read_csv("../input/train.csv")
train.Class.value_counts()
test = pd.read_csv("../input/test.csv")
num_classes = 3

from sklearn.preprocessing import LabelEncoder
import keras
lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)
train.Class.value_counts(normalize=True)
def create_model():
    model = Sequential()
#     model.add(Conv2D(32, (2,2), strides=(1,1), activation='relu', padding='valid', input_shape=(32, 32, 3)))
#     BatchNormalization()
#     model.add(Conv2D(32,(2,2),activation='relu'))
#     BatchNormalization()
#     model.add(MaxPool2D(pool_size=(2,2)))
#     model.add(Conv2D(64, (2,2), strides=(1,1), activation='relu', padding='valid'))
#     model.add(MaxPool2D(pool_size=(2,2)))
#     model.add(Conv2D(64, (2,2), strides=(1,1), activation='relu', padding='valid'))
#     model.add(Flatten())
#     model.add(Dense(512,activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return(model)
#======================================New Code==================================================================#
    model.add(Convolution2D(32, 5, 5, input_shape=(32, 32, 3), border_mode='same', activation = 'relu'))
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(120, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return(model)

    ############ New Code################################
  
# z = Sequential([
#         BatchNormalization(input_shape = (32,32,3)),
#         Conv2D(32,(5,5), activation='relu'),
#         BatchNormalization(),
#         Conv2D(32,(5,5), activation='relu'),
#         BatchNormalization(),
#         MaxPool2D(),
#         Conv2D(64,(5,5), activation='relu'),
#         BatchNormalization(),
#         Conv2D(64,(5,5), activation='relu'),
#         BatchNormalization(),
#         MaxPool2D(),
#         Flatten(),
#         Dense(512, activation='relu'),
#         Dense(3, activation='softmax')
#         ])
# z.compile(optimizer = 'nadam' , loss='categorical_crossentropy', metrics=['accuracy'])
z = create_model()
# z = Sequential([
#         BatchNormalization(input_shape = (32,32,3)),
#         Convolution2D(32,(3,3), activation='relu'),
# #         BatchNormalization(),
#         Convolution2D(32,(3,3), activation='relu'),
# #         BatchNormalization(),
#         MaxPooling2D(),
#         Convolution2D(64,(3,3), activation='relu'),
# #         BatchNormalization(),
#         Convolution2D(64,(3,3), activation='relu'),
# #         BatchNormalization(),
#         MaxPooling2D(),
#         Flatten(),
#         Dropout(0.3),
#         Dense(384, activation='relu'),
#         Dropout(0.6),
#         Dense(3, activation='softmax')
#         ])
z.compile(optimizer = 'nadam' , loss='categorical_crossentropy', metrics=['accuracy'])
z.summary()
train_x.shape
hist = z.fit(train_x, train_y,epochs=5,verbose=1, validation_split=0.40)
# visualizing losses and accuracy
%matplotlib inline

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

epochs = range(len(train_acc))

plt.plot(epochs,train_loss,'r', label='train_loss')
plt.plot(epochs,val_loss,'b', label='val_loss')
plt.title('train_loss vs val_loss')
plt.legend()
plt.figure()

plt.plot(epochs,train_acc,'r', label='train_acc')
plt.plot(epochs,val_acc,'b', label='val_acc')
plt.title('train_acc vs val_acc')
plt.legend()
plt.figure()
sample_submission = pd.read_csv(os.path.join('../input/', 'Sample_Submission.csv'))
test_x_temp = test_x.reshape(-3, 32, 32, 3)
pred = z.predict_classes(test_x_temp)

pred.shape
pred_f = lb.inverse_transform(pred)
sample_submission.ID = test.ID; sample_submission.Class = pred_f
sample_submission.to_csv(os.path.join(data_dir, 'sub.csv'), index=False)
sample_submission.shape
sample_submission