# Data preprocessing and visualization
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import glob

# Deep learning
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf 
from tensorflow.keras.preprocessing import image

# Deep learning helpers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
DATASET_DIR = "/kaggle/input/cat-and-dog/training_set/training_set"

os.listdir(DATASET_DIR)
dog_images = []

for img_path in glob.glob(DATASET_DIR + "/dogs/*"):
    if img_path != "/kaggle/input/cat-and-dog/training_set/training_set/dogs/_DS_Store": #ignore irrelevant data
        dog_images.append(mpimg.imread(img_path))
    
fig = plt.figure()
fig.suptitle("Dog")
plt.imshow(dog_images[0], cmap="gray")
plt.show()
cat_images = []

for img_path in glob.glob(DATASET_DIR + "/cats/*"):
    if img_path != "/kaggle/input/cat-and-dog/training_set/training_set/cats/_DS_Store":
        cat_images.append(mpimg.imread(img_path))
    
fig = plt.figure()
fig.suptitle("Cat")
plt.imshow(cat_images[0], cmap="gray")
plt.show()
print(str(len(dog_images))+" dog images")
print(str(len(cat_images))+" cat images")
IMG_W = 50
IMG_H = 50
CHANNELS = 3

INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)
NB_CLASSES = 2
EPOCHS = 24
BATCH_SIZE = 100
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf 
from tensorflow.keras.preprocessing import image

# All Parameters of Conv2D:
#-------------------------------------------------------
#tf.keras.layers.Conv2D(
#    filters,
#    kernel_size,
#    strides=(1, 1),
#    padding="valid",
#    data_format=None,
#    dilation_rate=(1, 1),
#    groups=1,
#    activation=None,
#    use_bias=True,
#    kernel_initializer="glorot_uniform",
#    bias_initializer="zeros",
#    kernel_regularizer=None,
#    bias_regularizer=None,
#    activity_regularizer=None,
#    kernel_constraint=None,
#    bias_constraint=None,
#    **kwargs
#)
#-------------------------------------------------------

# All Parameters of MaxPooling2D:
#-------------------------------------------------------
#tf.keras.layers.MaxPooling2D(
#    pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs
#)
#-------------------------------------------------------

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = INPUT_SHAPE, activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters = 48, kernel_size = (3,3), activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (1,1)))
model.add(Dropout(0.25))

#fully connected
model.add(Flatten())
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "sigmoid"))

# compile 
model.compile(loss = "binary_crossentropy",
             optimizer = "rmsprop",
             metrics = ["accuracy"])
model.summary()
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,
                                  validation_split = 0.25)

# All Parameters of ImageDataGenerator:
#-------------------------------------------------------
#tf.keras.preprocessing.image.ImageDataGenerator(      
#    featurewise_center=False,                         
#    samplewise_center=False,                            
#    featurewise_std_normalization=False,
#    samplewise_std_normalization=False,
#    zca_whitening=False,
#    zca_epsilon=1e-06,
#    rotation_range=0,
#    width_shift_range=0.0,
#    height_shift_range=0.0,
#    brightness_range=None,
#    shear_range=0.0,
#    zoom_range=0.0,
#    channel_shift_range=0.0,
#    fill_mode="nearest",
#    cval=0.0,
#    horizontal_flip=False,
#    vertical_flip=False,
#    rescale=None,
#    preprocessing_function=None,
#    data_format=None,
#    validation_split=0.0,
#    dtype=None,
#)
#--------------------------------------------------------
            
train_generator = train_datagen.flow_from_directory(
DATASET_DIR,
target_size = (IMG_H, IMG_W),
batch_size = BATCH_SIZE,
class_mode = "binary",
subset = "training")

validation_generator = train_datagen.flow_from_directory(
DATASET_DIR,
target_size = (IMG_H,IMG_W),
batch_size = BATCH_SIZE,
class_mode ="binary",
shuffle = False,
subset = "validation")

#fitting
hist = model.fit_generator(
train_generator,
steps_per_epoch = train_generator.samples//BATCH_SIZE,
validation_data = validation_generator,
validation_steps = validation_generator.samples // BATCH_SIZE,
epochs = EPOCHS)
model.save("cnn_cat-dog_classifier_v1.h5")
plt.figure(figsize = (13,7))
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc = "upper left")
#plt.text(23,0.5,"Current Training Accuracy: "+str(np.round(hist.history["accuracy"][-1]*100,2))+"%",fontsize = 18,color = "black")
#plt.text(23,0.46,"Current Validation Accuracy: "+str(np.round(hist.history["val_accuracy"][-1]*100,2))+"%",fontsize = 18,color = "black")
plt.show()
plt.figure(figsize = (13,7))
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc = "upper left")
#plt.text(26,0.8,"Current Training Loss: "+str(np.round(hist.history["loss"][-1],3)),fontsize = 18,color = "black")
#plt.text(26,0.73,"Current Validation Loss: "+str(np.round(hist.history["val_loss"][-1],3)),fontsize = 18,color = "black")
plt.show()
print("Training Accuracy: "+str(np.round(hist.history["accuracy"][-1]*100,2))+"%")
print("Validation Accuracy: "+str(np.round(hist.history["val_accuracy"][-1]*100,2))+"%")
label = validation_generator.classes
pred = model.predict(validation_generator)
predicted_class_indices = np.argmax(pred, axis = 1)
labels = (validation_generator.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
predictions = [labels2[k] for k in predicted_class_indices]
print(labels)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(predicted_class_indices, label)
cm
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm,annot = True, linewidths = 0.3,cmap = "Reds",annot_kws = {"size": 18}, linecolor = "black", fmt = ".0f", ax=ax )
plt.xlabel("Prediction")
plt.title("Confusion Matrix (Cat = 0, Dog = 1)")
plt.ylabel("True")
plt.show()
dog_images = []
count = 0

for img_path in glob.glob(DATASET_DIR + "/dogs/*"): # you can use enumerate() too
    count += 1
    dog_images.append(image.load_img(str(img_path), target_size = (150,150,3)))
    if count > 374:
        break
    
fig = plt.figure()
fig.suptitle("Dog")
plt.imshow(dog_images[15], cmap="gray")
plt.show()
cat_images = []
count = 0

for img_path in glob.glob(DATASET_DIR + "/cats/*"):
    count += 1
    cat_images.append(image.load_img(str(img_path), target_size = (150,150,3)))
    if count > 374:
        break
    
fig = plt.figure()
fig.suptitle("Cat")
plt.imshow(cat_images[20], cmap="gray")
plt.show()
dog_images_resized = []

for i in dog_images:
    dog_images_resized.append(i.resize((50,50)))
cat_images_resized = []

for i in cat_images:
    cat_images_resized.append(i.resize((50,50)))
print(str(len(dog_images_resized))+" dog images")
print(str(len(cat_images_resized))+" cat images")
images_together = []

for i in dog_images_resized:
    images_together.append(img_to_array(i))
    
for i in cat_images_resized:
    images_together.append(img_to_array(i))
    
targets = np.zeros(len(images_together))
targets[:len(dog_images_resized)-1] = 1 #dog-> 1, cat-> 0
print("image list length: ",len(images_together))
print("target list length: ",len(targets))
targets = np.array(targets)
print("targets: ",targets.shape)
targets = targets.reshape(-1,1)
print("new shape of targets: ",targets.shape)
images_together = np.array(images_together)
print("shape of images together: ",images_together.shape)
# we have to shuffle the data, I found that solution but if you know a better solution let me know in the comments
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images_together, targets, test_size=0.25, stratify=targets)

images_together = np.concatenate((X_train, X_val))
targets = np.concatenate((y_train, y_val))
IMG_W = 50
IMG_H = 50
CHANNELS = 3

INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)
NB_CLASSES = 2
EPOCHS = 24
BATCH_SIZE = 10
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Reshape
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf 
from tensorflow.keras.preprocessing import image

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = INPUT_SHAPE , activation = "relu")) #input_sahpe = INPUT_SHAPE
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters = 48, kernel_size = (3,3), activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (1,1)))
model.add(Dropout(0.25))

#fully connected
model.add(Flatten())
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "sigmoid"))

# compile 
model.compile(loss = "binary_crossentropy",
             optimizer = "rmsprop",
             metrics = ["accuracy"])
model.summary()
validation_split = 0.25
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,
                                  validation_split = validation_split)


train_generator = train_datagen.flow(
images_together, targets,
batch_size = BATCH_SIZE,
subset = "training")

validation_generator = train_datagen.flow(
images_together, targets,
batch_size = BATCH_SIZE,
shuffle = False,
subset = "validation")

#fitting
hist = model.fit_generator(
train_generator,
steps_per_epoch = (len(images_together)*(1-validation_split))//BATCH_SIZE,
validation_data = validation_generator,
validation_steps = (len(images_together)*validation_split)// BATCH_SIZE,
epochs = EPOCHS)
plt.figure(figsize = (13,7))
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc = "upper left")
#plt.text(23,0.5,"Current Training Accuracy: "+str(np.round(hist.history["accuracy"][-1]*100,2))+"%",fontsize = 18,color = "black")
#plt.text(23,0.46,"Current Validation Accuracy: "+str(np.round(hist.history["val_accuracy"][-1]*100,2))+"%",fontsize = 18,color = "black")
plt.show()
plt.figure(figsize = (13,7))
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc = "upper right")
#plt.text(26,0.8,"Current Training Loss: "+str(np.round(hist.history["loss"][-1],3)),fontsize = 18,color = "black")
#plt.text(26,0.73,"Current Validation Loss: "+str(np.round(hist.history["val_loss"][-1],3)),fontsize = 18,color = "black")
plt.show()
print("Training Accuracy: "+str(np.round(hist.history["accuracy"][-1]*100,2))+"%")
print("Validation Accuracy: "+str(np.round(hist.history["val_accuracy"][-1]*100,2))+"%")