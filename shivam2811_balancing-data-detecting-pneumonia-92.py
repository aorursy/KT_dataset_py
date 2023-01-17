import os
import glob
import tensorflow as tf
import h5py
import shutil
import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
#from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Input, Concatenate, Dropout
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import cv2
from glob import glob
from keras import backend as K
color = sns.color_palette()
%matplotlib inline
data_dir = "../input/chest-xray-pneumonia/chest_xray/"
train_dir = os.path.join(data_dir, "train/")
test_dir = os.path.join(data_dir, "test/")
val_dir = os.path.join(data_dir, "val/")
train_data = []

normal_cases_dir = os.path.join(train_dir,'NORMAL')
pneumonia_cases_dir = os.path.join(train_dir,'PNEUMONIA')

# Get the list of all the images
normal_cases = glob(normal_cases_dir+"/*.jpeg")
pneumonia_cases = glob(pneumonia_cases_dir + "/*.jpeg")

# Go through all the normal cases. The label for these cases will be 0
for img in normal_cases:
    train_data.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in pneumonia_cases:
    train_data.append((img, 1))

# Get a pandas dataframe from the data we have in our list 
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)

# How the dataframe looks like?
train_data.head()
cases_count = train_data['label'].value_counts()
print(cases_count)

# Plot the results 
plt.figure(figsize=(10,8))
sns.barplot(x=cases_count.index, y= cases_count.values)
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()
val_data = []

normal_cases_dir = os.path.join(val_dir,'NORMAL')
pneumonia_cases_dir = os.path.join(val_dir,'PNEUMONIA')

# Get the list of all the images
normal_cases = glob(normal_cases_dir+"/*.jpeg")
pneumonia_cases = glob(pneumonia_cases_dir + "/*.jpeg")

# Go through all the normal cases. The label for these cases will be 0
for img in normal_cases:
    val_data.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in pneumonia_cases:
    val_data.append((img, 1))

# Get a pandas dataframe from the data we have in our list 
val_data = pd.DataFrame(val_data, columns=['image', 'label'],index=None)

# How the dataframe looks like?
val_data.head()
cases_count = val_data['label'].value_counts()
print(cases_count)

# Plot the results 
plt.figure(figsize=(10,8))
sns.barplot(x=cases_count.index, y= cases_count.values)
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()
test_data = []

normal_cases_dir = os.path.join(test_dir,'NORMAL')
pneumonia_cases_dir = os.path.join(test_dir,'PNEUMONIA')

# Get the list of all the images
normal_cases = glob(normal_cases_dir+"/*.jpeg")
pneumonia_cases = glob(pneumonia_cases_dir + "/*.jpeg")

# Go through all the normal cases. The label for these cases will be 0
for img in normal_cases:
    test_data.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in pneumonia_cases:
    test_data.append((img, 1))

# Get a pandas dataframe from the data we have in our list 
test_data = pd.DataFrame(test_data, columns=['image', 'label'],index=None)

# How the dataframe looks like?
test_data.head()
cases_count = test_data['label'].value_counts()
print(cases_count)

# Plot the results 
plt.figure(figsize=(10,8))
sns.barplot(x=cases_count.index, y= cases_count.values)
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()
pneumonia_samples = (train_data[train_data['label']==1]['image'].iloc[:5]).tolist()
normal_samples = (train_data[train_data['label']==0]['image'].iloc[:5]).tolist()

# Concat the data in a single list and del the above two list
samples = pneumonia_samples + normal_samples
del pneumonia_samples, normal_samples

# Plot the data 
f, ax = plt.subplots(2,5, figsize=(30,10))
for i in range(10):
    img = imread(samples[i])
    ax[i//5, i%5].imshow(img, cmap='gray')
    if i<5:
        ax[i//5, i%5].set_title("Pneumonia")
    else:
        ax[i//5, i%5].set_title("Normal")
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
plt.show()
Normal = glob(train_dir + "NORMAL/*") + glob(test_dir + "NORMAL/*") + glob(val_dir + "NORMAL/*")
Pneumonia = glob(train_dir + "PNEUMONIA/*") + glob(test_dir + "PNEUMONIA/*") + glob(val_dir + "PNEUMONIA/*")

print("We have total {} Normal Images".format(len(Normal)))
print("We have total {} Pneumonia Images".format(len(Pneumonia)))

# For all normal images
Train_normal = Normal[:1266]
Test_normal = Normal[1266:1424]
Validation_normal = Normal[1424:]

# For all pneumonia images
Train_pneumonia = Pneumonia[:3418]
Test_pneumonia = Pneumonia[3418:3845]
Validation_pneumonia = Pneumonia[3845:]


!mkdir Train
!mkdir Test
!mkdir Validation
!mkdir Train/Normal
!mkdir Train/Pneumonia
!mkdir Test/Normal
!mkdir Test/Pneumonia
!mkdir Validation/Normal
!mkdir Validation/Pneumonia
# copy all images from Train_pneumonia to Train/Pneumonia
for i in Train_pneumonia:
    shutil.copy(i, "Train/Pneumonia/")
print("Copied all images from Train_pneumonia to Train/Pneumonia")
    
# copy all images from Test_pneumonia to Test/Pneumonia
for i in Test_pneumonia:
    shutil.copy(i, "Test/Pneumonia/")
print("Copied all images from Test_pneumonia to Test/Pneumonia")
    
# copy all images from Validation_pneumonia to Validation/Pneumonia
for i in Validation_pneumonia:
    shutil.copy(i, "Validation/Pneumonia/")
print("Copied all images from Validation_pneumonia to Validation/Pneumonia")

# copy all images from Train_normal to Train/Normal
for i in Train_normal:
    shutil.copy(i, "Train/Normal/")
print("Copied all images from Train_normal to Train/Normal")

# copy all images from Test_normal to Test/Normal
for i in Test_normal:
    shutil.copy(i, "Test/Normal/")
print("Copied all images from Test_normal to Test/Normal")
    
# copy all images from Validation_normal to Validation/Normal
for i in Validation_normal:
    shutil.copy(i, "Validation/Normal/")
print("Copied all images from Validation_normal to Validation/Normal")
train_dir = "Train/"
test_dir = "Test/"
val_dir = "Validation/"
# check new data distribution
print("Number of images in Train is {}".format(len(glob(train_dir + "*/*"))))
print("Number of images in Test is {}".format(len(glob(test_dir + "*/*"))))
print("Number of images in Validation is {}".format(len(glob(val_dir + "*/*"))))

filenames = tf.io.gfile.glob(str('Train/*/*'))
train_filenames=tf.io.gfile.glob(str('Train/*/*'))
filenames.extend(tf.io.gfile.glob(str('Validation/*/*')))
val_filenames=tf.io.gfile.glob(str('Validation/*/*'))

COUNT_NORMAL = len([filename for filename in train_filenames if "Normal" in filename])
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len([filename for filename in train_filenames if "Pneumonia" in filename])
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))
initial_bias = np.log([COUNT_PNEUMONIA/COUNT_NORMAL])
initial_bias
weight_for_0 = (1 / COUNT_NORMAL)*(len(train_filenames))/2.0 
weight_for_1 = (1 / COUNT_PNEUMONIA)*(len(train_filenames))/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

train_datagen = ImageDataGenerator(rotation_range = 30,
                                   zoom_range = 0.2,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   horizontal_flip = True,
                                   rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 16
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (224, 224), 
                                                 batch_size = batch_size, 
                                                 class_mode = "binary")
val_set = val_datagen.flow_from_directory(val_dir,
                                          target_size = (224, 224),
                                          batch_size = batch_size,
                                          class_mode = 'binary')
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size = (224, 224),
                                            batch_size = batch_size,
                                            class_mode = 'binary')
def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x
def transition_block(x, reduction, name):
    bn_axis = 3
    if K.image_data_format() == "channels_first":
        bn_axis = 1
    x = BatchNormalization(axis = bn_axis, epsilon=1.001e-5, name = name + '_bn')(x)
    x = Activation('relu', name = name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias = False, 
               name = name + '_conv')(x)
    x = AveragePooling2D(2, strides = 2, name = name + '_pool')(x)
    return x
def conv_block(x, growth_rate, name):
    bn_axis = 3
    if K.image_data_format() == "channels_first":
        bn_axis = 1
    
    x1 = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5,
                            name = name + '_0_bn')(x)
    x1 = Activation('relu', name = name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias = False,
                name = name + '_1_conv')(x1)
    
    x1 = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = name + '_1_bn')(x1)
    x1 = Activation('relu', name = name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding = 'same', use_bias = False,
                name = name + '_2_conv')(x1)
    
    x = Concatenate(axis = bn_axis, name = name + '_concat')([x, x1])
    return x
def DenseNet121(blocks, input_shape = (224, 224, 3), classes = 1):
    
    # Determine proper input shape
    img_input = Input(shape=input_shape)
    bn_axis = 3
    if K.image_data_format() == "channels_first":
        bn_axis = 1

    x = ZeroPadding2D(padding = ((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides = 2, use_bias = class_weight, name = 'conv1/conv')(x)
    x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = 'conv1/bn')(x)
    x = Activation('relu', name = 'conv1/relu')(x)
    x = ZeroPadding2D(padding = ((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides = 2, name = 'pool1')(x)

    x = dense_block(x, blocks[0], name = 'conv2')
    x = transition_block(x, 0.5, name = 'pool2')
    x = dense_block(x, blocks[1], name = 'conv3')
    x = transition_block(x, 0.5, name = 'pool3')
    x = dense_block(x, blocks[2], name = 'conv4')
    x = transition_block(x, 0.5, name = 'pool4')
    x = dense_block(x, blocks[3], name = 'conv5')

    x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = 'bn')(x)
    x = Activation('relu', name = 'relu')(x)
    
    basemodel = Model(img_input, x, name='basedensenet121')
    #basemodel.load_weights(weights="imagenet",)
    
    x = basemodel.output
    x = GlobalAveragePooling2D(name = 'avg_pool')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation = 'sigmoid', name = 'fc1')(x)
    
    model = Model(img_input, x, name='densenet121')
    return model

model = DenseNet121(blocks = [6, 12, 24, 16], input_shape = (224, 224, 3), classes = 1)
model.summary()
opt = Adam(lr = 0.001)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
# this will help in reducing learning rate by factor of 0.1 when accuarcy will not improve
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 2, verbose = 1,
                                            factor = 0.3, min_lr = 0.000001)
class myCallback(Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('val_accuracy') > 0.90):
			print("\nReached 90% accuracy so cancelling training!")
			self.model.stop_training = True

callbacks = myCallback()
H = model.fit_generator(training_set,
                        steps_per_epoch = training_set.samples//batch_size,
                        validation_data = val_set,
                        epochs = 20,
                        validation_steps = val_set.samples//batch_size,
                        callbacks = [reduce_lr, callbacks])
print("Loss of the model is - " , model.evaluate(test_set)[0])
print("Accuracy of the model is - " , model.evaluate(test_set)[1]*100 , "%")
