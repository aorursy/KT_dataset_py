import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os # accessing directory structure
import sys
import random
# checking a random picture from different categories
for i in range(10):
    random_train_class = random.choice(os.listdir('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train'))
    random_train_image = random.choice(os.listdir('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/'+random_train_class))
    print(random_train_class,random_train_image)
print("***************************************************************************************************************************")

img = plt.imread('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/'+random_train_class+'/'+random_train_image)
plt.imshow(img)
plt.title("random_train_class->"+random_train_class)
plt.show()

for i in range(10):
    random_valid_class = random.choice(os.listdir('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid'))
    random_valid_image = random.choice(os.listdir('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/'+random_train_class))
    print(random_valid_class,random_valid_image)
img = plt.imread('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/'+random_train_class+'/'+random_valid_image)
plt.imshow(img)
plt.title("random_valid_class->"+random_valid_class)

import tensorflow as tf
import keras_preprocessing
from tensorflow.keras.preprocessing import image
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
tf.__version__
# Any results you write to the current directory are saved as output.
TRAINING_DIR = '/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/' 
VALIDATION_DIR = '/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/'

# this is the augmentation configuration we will use for training
train_gen = ImageDataGenerator(rescale = 1./255)
valid_gen = ImageDataGenerator(rescale = 1./255)
 
#rgb --> the images will be converted to have 3 channels.

train_data = train_gen.flow_from_directory(
TRAINING_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb",
batch_size=128
)

valid_data = valid_gen.flow_from_directory(
VALIDATION_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb"
)
#
for cl_indis, cl_name in enumerate(train_data.class_indices):
     print(cl_indis, cl_name)
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization
# Initializing the CNN based AlexNet
model = Sequential()

#valid:zero padding, same:keep same dimensionality by add padding

# Convolution Step 1
model.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(227, 227, 3), activation = 'relu'))

# Max Pooling Step 1
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))

# Convolution Step 2
model.add(Convolution2D(256, 5, strides = (1, 1), padding='same', activation = 'relu'))

# Max Pooling Step 2
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding='valid'))


# Convolution Step 3
model.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))

# Convolution Step 4
model.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))


# Convolution Step 5
model.add(Convolution2D(256, 3, strides=(1,1), padding='same', activation = 'relu'))

# Max Pooling Step 3
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))

# Flattening Step --> 6*6*256 = 9216
model.add(Flatten())

# Full Connection Steps
# 1st Fully Connected Layer
model.add(Dense(units = 4096, activation = 'relu'))

# 2nd Fully Connected Layer
model.add(Dense(units = 4096, activation = 'relu'))

# 3rd Fully Connected Layer
model.add(Dense(units = 10, activation = 'softmax'))

model.summary()
from keras.optimizers import Adam
import keras
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#train_num/128 = 144
#valid_num//128 = 35
train_num = train_data.n
valid_num = valid_data.n

train_batch_size = train_data.batch_size # choose 128
valid_batch_size = valid_data.batch_size #default 32

STEP_SIZE_TRAIN = train_num//train_batch_size #144
STEP_SIZE_VALID = valid_num//valid_batch_size #144

history = model.fit_generator(generator=train_data,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_data,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25
)
#saving model
filepath="AlexNetModel.hdf5"
model.save(filepath)
#plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
from keras.preprocessing import image
import numpy as np
class_list = list(valid_data.class_indices.keys())
#class_dict = training_set.class_indices
#li = list(class_dict.keys())
#print(li)

for i in range(10):
    random_val_class = random.choice(os.listdir('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid'))
    random_val_image = random.choice(os.listdir('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/'+random_val_class))
    image_path = '/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/'+random_val_class+'/'+random_val_image
    
    #prepocessing
    new_img = image.load_img(image_path, target_size=(227, 227))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    
    prediction = model.predict(img)
    d = prediction.flatten()
    j = d.max()
    for index,item in enumerate(d):
        if item == j:
            class_name = class_list[index]
            print("{0}\nGround Class: {1}                      Predict Class:{2}".format(random_val_image,random_val_class,class_name))
            print("****************************************************************************************************************")

#TO UNDERSTAND DATA

#train_data is a 'DirectoryIterator' yielding tuples of (X, y). 
#We divide the dataset into 128 pieces(batch). This means that we can pass the dataset over 144 times in total.
print(len(train_data)) #144 is total step count
print("**********************************************************")
print(len(train_data[0])) # this is a tupple (X,y)-->; s is step count, s=0,1,2...143 (so this a tupple at 0. step)
print("**********************************************************")
print(len(train_data[0][0])) # return lists of [X] --> The lenth is (128x227,227,3), 128 is batch size, (227,227) is input size of image and 3 is channel number(RGB)
print(train_data[0][0].shape)
print("**********************************************************")
print(len(train_data[0][0][0])) #X at i. batch --> (227,227,3)
print(train_data[0][0][0].shape)
print("**********************************************************")
print(train_data[0][0][0][0].shape) # return a row
plt.imshow(train_data[0][0][0][5]) #plot 5.row
print("**********************************************************")
# we plot between 0-50 row
plt.imshow(train_data[0][0][0][0:50])
plt.show()

print(train_data[0][1].shape) # y, The lenth is (128,10)
print("**********************************************************")
print(train_data[0][1][0]) #get 0.picture label
print(train_data[0][1][3]) #get 3rd picture label
def f_class_by_array(cl_arr):
    cl = 0
    for i in range(len(cl_arr)):
        if cl_arr[i] == 1:
            cl_name = f_class_name_by_cl(cl)
            return cl_name
        else:
            cl += 1
            
def f_class_name_by_cl(cl):
    for cl_name, cl_ind in train_data.class_indices.items():
        if cl_ind == cl:
            return cl_name
plt.figure(figsize=(20,10))
for i in range(5): #first 5 images
    plt.subplot(5/5+1, 5, i+1)
    cl_name = f_class_by_array(train_data[i][1][0])
    plt.title("{}".format(cl_name))
    plt.imshow(train_data[i][0][0])


