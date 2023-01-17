# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for filenames in os.listdir('/kaggle/input/weed-detection-in-soybean-crops/dataset'):
    print(filenames)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
### TRAIN,TEST AND VALIDATION SPILIT
import os
import os
import shutil
import numpy as np

base_dir = '/kaggle/working/segmented_image'
os.makedirs(base_dir, exist_ok=True)


train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

train_grass_dir = os.path.join(train_dir, 'grass')
os.makedirs(train_grass_dir, exist_ok=True)

train_soil_dir = os.path.join(train_dir, 'soil')
os.makedirs(train_soil_dir, exist_ok=True)

train_soybean_dir = os.path.join(train_dir, 'soybean')
os.makedirs(train_soybean_dir, exist_ok=True)

train_weed_dir = os.path.join(train_dir, 'weed')
os.makedirs(train_weed_dir, exist_ok=True)


validation_grass_dir = os.path.join(validation_dir, 'grass')
os.makedirs(validation_grass_dir, exist_ok=True)

validation_soil_dir = os.path.join(validation_dir, 'soil')
os.makedirs(validation_soil_dir, exist_ok=True)

validation_soybean_dir = os.path.join(validation_dir, 'soybean')
os.makedirs(validation_soybean_dir, exist_ok=True)

validation_weed_dir = os.path.join(validation_dir, 'weed')
os.makedirs(validation_weed_dir, exist_ok=True)


test_grass_dir = os.path.join(test_dir, 'grass')
os.makedirs(test_grass_dir, exist_ok=True)

test_soil_dir = os.path.join(test_dir, 'soil')
os.makedirs(test_soil_dir, exist_ok=True)

test_soybean_dir = os.path.join(test_dir, 'soybean')
os.makedirs(test_soybean_dir, exist_ok=True)

test_weed_dir = os.path.join(test_dir, 'weed')
os.makedirs(test_weed_dir, exist_ok=True)
## DEFINING THE DIRECTORY FOR THE ORIGINAL DIRECTORY

original_dir_weed ='/kaggle/input/weed-detection-in-soybean-crops/dataset/broadleaf'
original_dir_grass = '/kaggle/input/weed-detection-in-soybean-crops/dataset/grass'
original_dir_soil = '/kaggle/input/weed-detection-in-soybean-crops/dataset/soil'
original_dir_soybean = '/kaggle/input/weed-detection-in-soybean-crops/dataset/soybean'

### SEGMENTING THE TRAIN, VALIDATION AND TEST IMAGES
### TRAINING DATESET WOULD HAVE 70 PERCENT OF DATASET AND 15 PERCENT FOR VALIDATION & TEST
soil_total_images = len(os.listdir(original_dir_soil))
weed_total_images = len(os.listdir(original_dir_weed))
grass_total_images = len(os.listdir(original_dir_grass))
soybean_total_images = len(os.listdir(original_dir_soybean))

### COPYING 70 PERCENT OF SOIL DATASET TO SOIL TRAIN FOLDER
fnames = ['{}.tif'.format(i) for i in range(1,int(np.ceil(soil_total_images*0.70)))]
for fname in fnames:
    src = os.path.join(original_dir_soil, fname)
    dst = os.path.join(train_soil_dir, fname)
    shutil.copyfile(src, dst)

### COPYING THE 15 PERCENT OF SOIL DATASET TO SOIL VALIDATION FOLDER
fnames= ['{}.tif'.format(i) for i in range(int(np.ceil(soil_total_images*0.70)), int(np.ceil(soil_total_images*0.85)))]
for fname in fnames:
    src = os.path.join(original_dir_soil, fname)
    dst = os.path.join(validation_soil_dir, fname)
    shutil.copyfile(src, dst)
    
# COPYING 15 PERCENT OF SOIL DATASET TO SOIL TEST FOLDER
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(soil_total_images*0.85)), int(np.ceil(soil_total_images)))]
for fname in fnames:
    src = os.path.join(original_dir_soil, fname)
    dst = os.path.join(test_soil_dir, fname)
    shutil.copyfile(src, dst)
    
####SOYBEAN####
### COPYING 70 PERCENT OF SOIL DATASET TO SOYBEAN TRAIN FOLDER
fnames = ['{}.tif'.format(i) for i in range(1,int(np.ceil(soybean_total_images*0.70)))]
for fname in fnames:
    src = os.path.join(original_dir_soybean, fname)
    dst = os.path.join(train_soybean_dir, fname)
    shutil.copyfile(src, dst)

### COPYING THE 15 PERCENT OF SOYBEAN DATASET TO SOYBEAN VALIDATION FOLDER
fnames =  ['{}.tif'.format(i) for i in range(int(np.ceil(soybean_total_images*0.70)), int(np.ceil(soybean_total_images*0.85)))]
for fname in fnames:
    src = os.path.join(original_dir_soybean, fname)
    dst = os.path.join(validation_soybean_dir, fname)
    shutil.copyfile(src, dst)
    
# COPYING 15 PERCENT OF SOYBEAN DATASET TO SOYBEAN TEST FOLDER
fnames =  ['{}.tif'.format(i) for i in range(int(np.ceil(soybean_total_images*0.85)), soybean_total_images)]
for fname in fnames:
    src = os.path.join(original_dir_soybean, fname)
    dst = os.path.join(test_soybean_dir, fname)
    shutil.copyfile(src, dst)
    
####GRASS####
### COPYING 70 PERCENT OF GRASS DATASET TO GRASS TRAIN FOLDER
fnames = ['{}.tif'.format(i) for i in range(1,int(np.ceil(grass_total_images*0.70)))]
for fname in fnames:
    src = os.path.join(original_dir_grass, fname)
    dst = os.path.join(train_grass_dir, fname)
    shutil.copyfile(src, dst)

### COPYING THE 15 PERCENT OF GRASS DATASET TO GRASS VALIDATION FOLDER
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(grass_total_images*0.70)), int(np.ceil(grass_total_images*0.85)))]
for fname in fnames:
    src = os.path.join(original_dir_grass, fname)
    dst = os.path.join(validation_grass_dir, fname)
    shutil.copyfile(src, dst)
    
### COPYING THE 15 PERCENT OF GRASS DATASET TO GRASS TEST FOLDER
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(grass_total_images*0.85)), int(np.ceil(grass_total_images)))]
for fname in fnames:
    src = os.path.join(original_dir_grass, fname)
    dst = os.path.join(test_grass_dir, fname)
    shutil.copyfile(src, dst)
    
####WEED####
### COPYING THE 70 PERCENT OF WEED DATASET TO WEED TRAIN FOLDER
fnames = ['{}.tif'.format(i) for i in range(1,int(np.ceil(weed_total_images*0.70)))]
for fname in fnames:
    src = os.path.join(original_dir_weed, fname)
    dst = os.path.join(train_weed_dir, fname)
    shutil.copyfile(src, dst)

### COPYING THE 15 PERCENT OF WEED DATASET TO WEED VALIDATION FOLDER
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(weed_total_images*0.70)), int(np.ceil(weed_total_images*0.85)))]
for fname in fnames:
    src = os.path.join(original_dir_weed, fname)
    dst = os.path.join(validation_weed_dir, fname)
    shutil.copyfile(src, dst)
    
# Copiamos las siguientes 238 a test_weed_dir
fnames = ['{}.tif'.format(i) for i in range(int(np.ceil(weed_total_images*0.85)), weed_total_images)]
for fname in fnames:
    src = os.path.join(original_dir_weed, fname)
    dst = os.path.join(test_weed_dir, fname)
    shutil.copyfile(src, dst)
#### MODEL DEVELOPMENT
import os
import numpy as np
import keras
keras.__version__
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt

train_dir = "/kaggle/working/segmented_image/train"
test_dir = "/kaggle/working/segmented_image/test"
validation_dir  = "/kaggle/working/segmented_image/validation"

"""Parameters"""

img_width, img_height = 150, 150
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 128
nb_filters4 = 128

conv1_size = 3
conv2_size = 2
conv3_size = 2
conv4_size = 2
pool_size = 2
classes_num = 4
lr = 0.0004
dropout_value = 0.5

model = Sequential()

model.add(Convolution2D(nb_filters1, conv1_size, 
                        conv1_size, border_mode ="same",
                        input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters3, conv3_size, conv3_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512))

model.add(Activation("relu"))

model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

#ImageDataGenerator generates batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).
#Rather than performing the operations on your memory, the API is designed to be iterated by the deep learning model fitting process, 
#creating augmented image data for you just-in-time.

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=92,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=31,
        class_mode='categorical')

training_samples = sum(len(files) for _, _, files in os.walk(train_dir))
batch_size_training_generator=92
validation_samples =sum(len(files) for _, _, files in os.walk(validation_dir))
batch_size_validation_generator=31

model_verbosity = model.fit_generator(
      train_generator,
      steps_per_epoch=np.ceil(training_samples/batch_size_training_generator),
      epochs=15,
      validation_data=validation_generator,
      validation_steps=np.ceil(validation_samples/batch_size_validation_generator))


### SAVING THE FINAL MODEL IN THE LOCAL FILE
os.makedirs("kaggle/working/final_out",exist_ok = True )
model.save('kaggle/working/final_out/model_weedcrops.model')


acc = model_verbosity.history['accuracy']
val_acc = model_verbosity.history['val_accuracy']
loss = model_verbosity.history['loss']
val_loss = model_verbosity.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Training accuracy',color = 'black')
plt.title('Training Accuracy Chart')
plt.xlabel("Number Of Epochs")
plt.ylabel("Model Accuracy")
plt.legend()
plt.savefig("kaggle/working/final_out/training_accuracy_chart.jpeg")

plt.figure()
plt.plot(epochs, val_acc, label='Validation accuracy',color = 'black')
plt.title('Validation Accuracy Chart')
plt.xlabel("Number Of Epochs")
plt.ylabel("Model Accuracy")
plt.legend()
plt.savefig("kaggle/working/final_out/validation_accuracy_chart.jpeg")

plt.figure()
plt.plot(epochs, loss, label='Training loss',color = 'black')
plt.title('Training Loss Chart')
plt.xlabel("Number Of Epochs")
plt.ylabel("Loss Value")
plt.legend()
plt.savefig("kaggle/working/final_out/training_loss_chart.jpeg")

plt.figure()
plt.plot(epochs, val_loss, label='Validation Loss',color = 'black')
plt.title('Validation Loss Chart')
plt.xlabel("Number Of Epochs")
plt.ylabel("Loss Value")
plt.legend()
plt.savefig("kaggle/working/final_out/validation_loss_chart.jpeg")

plt.show()
for i in os.listdir("segmented_image"):
    print(i)
### IMAGE VALIDATION
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
from skimage import transform


test_dir = "segmented_image/test"

classifier = models.load_model("kaggle/working/final_out/model_weedcrops.model")

test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=31,
        class_mode='categorical')

test_samples = sum(len(files) for _, _, files in os.walk(test_dir))

batch_size_test=31

score= classifier.evaluate_generator(test_generator, 
                                steps = np.ceil(test_samples/batch_size_test))

print("\nTest accuracy for the CNN classifier : %.1f%%" % (100.0 * score[1]))



#### PREDICTING FOR SINGLE IMAGE

## IMAGE PRE PROCESSINF FUNCTION TO LOAD THE IMAGE DATA IN THE MODEL

def load(filename):
    np_image = Image.open(filename) #Open the image
    np_image = np.array(np_image).astype('float32')/255 
    np_image = transform.resize(np_image, (150, 150, 3)) 
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

label_map = (test_generator.class_indices)

print (label_map)

image_dir = "segmented_image/test/weed/1014.tif"
image_to_predict = load(image_dir)
result = classifier.predict(image_to_predict)
result= np.around(result,decimals=3)
result=result*100

print (result)
